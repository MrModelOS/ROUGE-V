// Канонические ldmatrix/stmatrix: roundtrip и transpose 8x8.
//
// ldmatrix_tile: 8x8 f16-тайл через ldmatrix.x1 + stmatrix.x1 обязан вернуться
// побитово. ldmatrix_trans: тот же тайл через .trans обязан транспонироваться.
// Значения in[r][c] = r*8+c (все точные в f16), сверка через f32.

#include "ptx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

static std::unique_ptr<rouge::PtxProgram> load_prog(const std::string& dir) {
  const std::string path = dir + "/ldmatrix_tile.ptx";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", path.c_str());
    return nullptr;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)n, '\0');
  if (std::fread(&text[0], 1, text.size(), f) != text.size()) {
    std::printf("FAIL: short read\n");
    return nullptr;
  }
  std::fclose(f);
  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) std::printf("FAIL parse: %s\n", err.c_str());
  return prog;
}

static int fn_index(const std::unique_ptr<rouge::PtxProgram>& prog, const char* name) {
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == name) return (int)i;
  return -1;
}

static bool run_entry(const rouge::PtxProgram& prog, int fi, float* out) {
  unsigned long long dev = reinterpret_cast<unsigned long long>(out);
  std::vector<uint8_t> blob(8, 0);
  std::memcpy(blob.data(), &dev, 8);
  std::string err;
  if (!rouge::execute_kernel(prog, fi, blob, 1, 1, 1, 32, 1, 1, &err, nullptr)) {
    std::printf("FAIL exec: %s\n", err.c_str());
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  auto prog = load_prog(dir);
  if (!prog) return 0;
  const int fi = fn_index(prog, "ldmatrix_tile");
  const int ft = fn_index(prog, "ldmatrix_trans");
  const int fs = fn_index(prog, "ldmatrix_sm75");
  const int fm = fn_index(prog, "stmatrix_sm90");
  EXPECT_TRUE(fi >= 0 && ft >= 0 && fs >= 0 && fm >= 0);
  if (fi < 0 || ft < 0 || fs < 0 || fm < 0) return 1;

  // Plain roundtrip: out[tid*2+e] == tid-row values row*8+col.
  {
    float out[64] = {0};
    EXPECT_TRUE(run_entry(*prog, fi, out));
    int bad = 0;
    for (int tid = 0; tid < 32; ++tid) {
      const int row = tid >> 2, col0 = (tid % 4) * 2;
      for (int e = 0; e < 2; ++e) {
        const float want = (float)(row * 8 + col0 + e);
        if (out[tid * 2 + e] != want) {
          if (bad < 5)
            std::printf("roundtrip tid=%d e=%d: got=%g want=%g\n", tid, e,
                        (double)out[tid * 2 + e], (double)want);
          ++bad;
        }
      }
    }
    EXPECT_TRUE(bad == 0);
  }

  // Transposed load: thread tid holds column tid>>2, rows (tid%4)*2..+1,
  // stored back to the same row layout, so out[tid*2+e] == in[col][row+e]
  // with col = tid>>2, row = (tid%4)*2.
  {
    float out[64] = {0};
    EXPECT_TRUE(run_entry(*prog, ft, out));
    int bad = 0;
    for (int tid = 0; tid < 32; ++tid) {
      const int col = tid >> 2, row0 = (tid % 4) * 2;
      for (int e = 0; e < 2; ++e) {
        // in[r][c] = r*8+c; we read column `col', rows row0+e.
        const float want = (float)((row0 + e) * 8 + col);
        if (out[tid * 2 + e] != want) {
          if (bad < 5)
            std::printf("trans tid=%d e=%d: got=%g want=%g\n", tid, e,
                        (double)out[tid * 2 + e], (double)want);
          ++bad;
        }
      }
    }
    EXPECT_TRUE(bad == 0);
  }

  // sm_75/sm_90 halves (ldmatrix-only / stmatrix-only): same x1 take
  // positions (row tid>>2, cols (tid%4)*2), checked like the roundtrip.
  for (const auto& ec : {std::make_pair(fs, "ldmatrix_sm75"),
                         std::make_pair(fm, "stmatrix_sm90")}) {
    float out[64] = {0};
    EXPECT_TRUE(run_entry(*prog, ec.first, out));
    int bad = 0;
    for (int tid = 0; tid < 32; ++tid) {
      const int row = tid >> 2, col0 = (tid % 4) * 2;
      for (int e = 0; e < 2; ++e) {
        const float want = (float)(row * 8 + col0 + e);
        if (out[tid * 2 + e] != want) {
          if (bad < 5)
            std::printf("%s tid=%d e=%d: got=%g want=%g\n", ec.second, tid, e,
                        (double)out[tid * 2 + e], (double)want);
          ++bad;
        }
      }
    }
    EXPECT_TRUE(bad == 0);
  }

  // x4: four matrices; EVERY thread gets a fragment of every matrix
  // (PTX ISA 9.7.16.5.15). Thread tid holds matrix k, row tid>>2, columns
  // (tid%4)*2 and +1, written out at out[k*64 + tid*2 + e].
  {
    const int fx = fn_index(prog, "ldmatrix_x4");
    EXPECT_TRUE(fx >= 0);
    if (fx >= 0) {
      float out[256] = {0};
      EXPECT_TRUE(run_entry(*prog, fx, out));
      int bad = 0;
      for (int tid = 0; tid < 32; ++tid) {
        const int row = tid >> 2, col0 = (tid % 4) * 2;
        for (int k = 0; k < 4; ++k) {
          for (int e = 0; e < 2; ++e) {
            const float want = (float)(k * 128 + 100 + row * 8 + col0 + e);
            const float got = out[k * 64 + tid * 2 + e];
            if (got != want) {
              if (bad < 6)
                std::printf("x4 tid=%d m=%d e=%d: got=%g want=%g\n", tid, k, e,
                            (double)got, (double)want);
              ++bad;
            }
          }
        }
      }
      EXPECT_TRUE(bad == 0);
    }
  }

  if (failures == 0)
    std::printf("PASS: ldmatrix/stmatrix roundtrip + transpose + x4\n");
  return failures ? 1 : 0;
}
