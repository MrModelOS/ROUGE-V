// Разбор nvFatbin: контейнер, который nvcc встраивает в объектный файл и
// передаёт нам в __cudaRegisterFatBinary.
//
// Формат восстановлен по байтам настоящего образа (nvcc 13.4, -arch=sm_75),
// а не по памяти. Один модуль выглядит так:
//
//   .nvFatBinSegment (дескриптор, его и передают как handle):
//       +0x00 u32  magic 0x466243B1
//       +0x04 u32  версия дескриптора
//       +0x08 u64  адрес контейнера
//   контейнер (.nv_fatbin):
//       +0x00 u32  magic 0xBA55ED50        (FATBIN_MAGIC)
//       +0x04 u16  version = 1
//       +0x06 u16  headerSize = 16
//       +0x08 u64  fatSize
//       далее записи по 64 байта, каждая с полезной нагрузкой сразу после:
//         +0x00 u16 kind (1 = PTX, 2 = cubin/ELF)
//         +0x08 u64 размер полезной нагрузки
//         +0x1A u16 smVersion (напр. 0x4B = sm_75)
//         +0x38 u64 размер PTX после распаковки (для kind == 1)
//       полезная нагрузка kind == 1:
//         +0x00 u64 служебное значение
//         +0x08 u64 служебное значение
//         +0x10 кадр Zstandard (magic 28 B5 2F FD)
//
// Полезная нагрузка kind == 2 — это ELF-cubin: машинный код для карты, он
// нам не нужен и не исполняется. Мы читаем только PTX.
//
// Распаковка — libzstd через dlopen: опциональная зависимость. Если её нет в
// системе, путь fatbin честно отказывает, а запуск продолжает работать через
// ROUGE_CUDA_PTX.

#include "fatbin.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kFatbinMagic = 0xBA55ED50u;   // "P.U" в hexdump
constexpr uint32_t kSegmentMagic = 0x466243B1u;
constexpr size_t kEntryHeaderSize = 64;
constexpr uint16_t kEntryKindPtx = 1;

template <typename T>
bool load(const uint8_t* base, size_t size, size_t offset, T* out) {
  if (offset > size || sizeof(T) > size - offset) return false;
  std::memcpy(out, base + offset, sizeof(T));
  return true;
}

// ---- optional libzstd -------------------------------------------------------

struct Zstd {
  void* handle = nullptr;
  unsigned long long (*getFrameContentSize)(const void*, size_t) = nullptr;
  size_t (*decompressStream)(void*, void*, void*) = nullptr;  // ZSTD_DCtx*, in, out
  void* (*createDStream)(void) = nullptr;                  // ZSTD_DStream*
  size_t (*freeDStream)(void*) = nullptr;
  const char* (*getErrorName)(size_t) = nullptr;

  bool ok() const {
    return handle != nullptr && getFrameContentSize != nullptr &&
           decompressStream != nullptr && createDStream != nullptr && freeDStream != nullptr;
  }
  static Zstd load_library() {
    Zstd z;
    for (const char* name : {"libzstd.so.1", "libzstd.so"}) {
      z.handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (z.handle != nullptr) break;
    }
    if (z.handle == nullptr) return z;
    z.getFrameContentSize =
        reinterpret_cast<decltype(z.getFrameContentSize)>(dlsym(z.handle, "ZSTD_getFrameContentSize"));
    z.decompressStream =
        reinterpret_cast<decltype(z.decompressStream)>(dlsym(z.handle, "ZSTD_decompressStream"));
    z.createDStream =
        reinterpret_cast<decltype(z.createDStream)>(dlsym(z.handle, "ZSTD_createDStream"));
    z.freeDStream = reinterpret_cast<decltype(z.freeDStream)>(dlsym(z.handle, "ZSTD_freeDStream"));
    z.getErrorName = reinterpret_cast<decltype(z.getErrorName)>(dlsym(z.handle, "ZSTD_getErrorName"));
    return z;
  }
};

// ZSTD_DCtx / ZSTD_inBuffer / ZSTD_outBuffer layout per the zstd.h ABI
// (stable, documented in the zstd API reference).
struct DStreamIn {
  const void* src;
  size_t size;
  size_t pos;
};
struct DStreamOut {
  void* dst;
  size_t size;
  size_t pos;
};
std::string decompress_zstd(const uint8_t* src, size_t size, size_t expected, std::string* err) {
  static const Zstd zstd = Zstd::load_library();
  if (!zstd.ok()) {
    *err = "libzstd is not available (needed to unpack PTX from a fatbin)";
    return std::string();
  }
  const unsigned long long frameSize =
      zstd.getFrameContentSize(src, size);
  const size_t hint = expected != 0 ? expected
                                     : (frameSize != 0 && frameSize < (1ull << 31)
                                            ? static_cast<size_t>(frameSize)
                                            : size_t(1) << 20);
  std::string out;
  out.resize(hint);

  // ZSTD_createDStream allocates the decompression context. (ZSTD_initDStream
  // is NOT a substitute: it initialises a context the caller already owns, and
  // its struct is opaque — passing a pointer-to-pointer there corrupts the
  // stack, which ASAN pointed out immediately.)
  void* ctx = zstd.createDStream();
  if (ctx == nullptr) {
    *err = "ZSTD_createDStream failed";
    return std::string();
  }
  DStreamIn in{src, size, 0};
  DStreamOut dst{&out[0], hint, 0};
  // zstd status codes: 0 = frame done, non-zero < 128 = "more input/output
  // expected"; anything at or above 128 is an error code to be reported.
  size_t last = 1;
  while (last != 0 && in.pos < in.size) {
    if (dst.pos == dst.size) {  // frame declared no size: keep growing
      dst.size *= 2;
      out.resize(dst.size);
      dst.dst = &out[0];
    }
    last = zstd.decompressStream(ctx, &dst, &in);
    if (last >= 128) {
      *err = std::string("zstd: ") +
             (zstd.getErrorName != nullptr ? zstd.getErrorName(last) : "decompression failed");
      zstd.freeDStream(ctx);
      return std::string();
    }
  }
  zstd.freeDStream(ctx);
  out.resize(dst.pos);
  return out;
}

}  // namespace

std::vector<std::string> rouge_fatbin_extract_ptx(const void* fatCubinHandle,
                                                   std::string* err) {
  std::vector<std::string> out;
  if (fatCubinHandle == nullptr) {
    *err = "no fatbin handle";
    return out;
  }
  // handle -> descriptor -> container
  const uint8_t* seg = static_cast<const uint8_t*>(fatCubinHandle);
  uint32_t segMagic = 0;
  uint64_t container = 0;
  if (!load(seg, 64, 0, &segMagic) || !load(seg, 64, 8, &container)) {
    *err = "fatbin descriptor is truncated";
    return out;
  }
  if (segMagic != kSegmentMagic) {
    // Not the descriptor layout: fall back to treating the handle as the
    // container itself, which is what some loaders hand over.
    container = reinterpret_cast<uint64_t>(fatCubinHandle);
    seg = static_cast<const uint8_t*>(fatCubinHandle);
    segMagic = kFatbinMagic;
  }

  const uint8_t* base = reinterpret_cast<const uint8_t*>(container);
  uint32_t magic = 0;
  uint16_t version = 0, headerSize = 0;
  uint64_t fatSize = 0;
  if (!load(base, 64, 0, &magic) || !load(base, 64, 4, &version) ||
      !load(base, 64, 6, &headerSize) || !load(base, 64, 8, &fatSize)) {
    *err = "fatbin container header is truncated";
    return out;
  }
  if (magic != kFatbinMagic) {
    *err = "not an nvFatbin container (magic is not 0xBA55ED50)";
    return out;
  }
  if (version != 1) {
    *err = "unsupported fatbin version " + std::to_string(version);
    return out;
  }
  if (headerSize < 16) {
    *err = "implausible fatbin header size";
    return out;
  }

  // fatSize counts the entries region AFTER the container header, so the
  // usable span is [headerSize, headerSize + fatSize). Entry payloads start
  // immediately after their 64-byte header with no padding, which is what the
  // byte layout of a real image shows (cubin payload ends exactly at the next
  // entry header).
  const size_t limit = headerSize + static_cast<size_t>(fatSize);
  size_t off = headerSize;
  while (off + kEntryHeaderSize <= limit) {
    uint16_t kind = 0;
    uint64_t payloadSize = 0;
    uint64_t uncompressed = 0;
    if (!load(base, limit, off, &kind) || !load(base, limit, off + 8, &payloadSize) ||
        !load(base, limit, off + 0x38, &uncompressed)) {
      *err = "fatbin entry header is truncated";
      return out;
    }
    const size_t payload = off + kEntryHeaderSize;
    if (payload > limit) {
      *err = "fatbin entry header runs past the container";
      return out;
    }
    if (payloadSize > limit - payload) {
      // A truncated tail is tolerated only for the last entry: the container's
      // declared size can exceed what the linker actually emitted.
      if (payloadSize - (limit - payload) > 16) {
        *err = "fatbin entry payload runs past the container";
        return out;
      }
      payloadSize = limit - payload;
    }
    if (kind == kEntryKindPtx) {
      // +0x10 inside the payload is the Zstandard frame (16 bytes of prefix).
      //
      // The declared payload size is NOT the compressed length: on a real image
      // the frame runs past it (entry says 400, the frame occupies 399 bytes
      // from +16, i.e. 415 in total). The frame is self-describing, so the
      // stream is handed everything up to the end of the container and the
      // streaming decoder stops on the frame boundary by itself.
      if (payload + 16 > limit) {
        *err = "PTX entry is truncated";
        return out;
      }
      std::string e;
      const std::string text =
          decompress_zstd(base + payload + 16, limit - (payload + 16),
                          static_cast<size_t>(uncompressed), &e);
      if (!e.empty()) {
        *err = e;
        return out;
      }
      if (!text.empty()) out.push_back(text);
    }
    off = payload + payloadSize;
  }
  if (out.empty()) *err = "no PTX entry in the fatbin";
  return out;
}