// rouge-cc — drop-in замена nvcc поверх ROUGE-V toolchain.
//
// Честная архитектура (юридически чистая):
//   .cu  -> настоящий пользовательский `nvcc` (чёрный ящик, только вызов
//           через PATH) делает .cu -> .ptx; затем наш ptx2ir + clang
//           делают .ptx -> объект под выбранную --target цель.
//   .ptx -> напрямую ptx2ir + clang, nvcc НЕ нужен.
// У нас НЕТ фронтенда CUDA C -> PTX: если nvcc нет в PATH, а вход .cu —
// честная ошибка, а не выдуманная компиляция.
//
// Поддерживаемое подмножество CLI nvcc:
//   -arch=sm_XX | -arch sm_XX | --gpu-architecture=sm_XX | -arch=native
//   -ptx, -c, -o <file>, -I<dir>/-I <dir>, -L<dir>/-L <dir>,
//   -D<mac>[=val]/-D <mac>, -O0/-O1/-O2/-O3, -g,
//   --target <triple> | --target=<triple> (расширение ROUGE-V:
//       host | riscv | amd | nvptx либо полный triple),
//   входные .cu / .ptx (плюс .ll / .o / .a для линковки),
//   -v/--verbose, -n/--dry-run, -h/--help, --version.
//
// Только C++17 + POSIX fork+exec, без внешних зависимостей.

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string target = "x86_64-pc-linux-gnu";  // канонический triple ptx2ir
  std::string smArch = "sm_75";                // из -arch=sm_XX (для nvptx)
  bool emitPtx = false;                        // -ptx
  bool compileOnly = false;                    // -c
  std::string output;                          // -o <file>
  std::vector<std::string> includes;           // -I (для nvcc и clang)
  std::vector<std::string> libdirs;            // -L (для линка)
  std::vector<std::string> defines;            // -D (для nvcc и clang)
  std::string optLevel = "-O2";                // -O0..-O3
  bool debugInfo = false;                      // -g
  bool verbose = false;                        // -v
  bool dryRun = false;                         // -n / --dry-run
  std::vector<std::string> inputs;
};

// Алиасы --target -> канонический triple ptx2ir. Пусто = неизвестно.
std::string canonical_target(const std::string& t) {
  if (t == "host" || t == "x86_64" || t == "x86_64-pc-linux-gnu") return "x86_64-pc-linux-gnu";
  if (t == "riscv" || t == "riscv64" || t == "riscv64-unknown-elf") return "riscv64-unknown-elf";
  if (t == "amd" || t == "amdgcn" || t == "amdgcn-amd-amdhsa") return "amdgcn-amd-amdhsa";
  if (t == "nvptx" || t == "nvptx64" || t == "nvptx64-nvidia-cuda") return "nvptx64-nvidia-cuda";
  return "";
}

// Флаги clang-бэкенда под triple (как в rouge_run_main.cpp).
// sm — "sm_XX" для nvptx (default sm_75).
std::vector<std::string> backend_flags(const std::string& target, const std::string& sm) {
  if (target.rfind("riscv", 0) == 0) return {"--target=riscv64-unknown-elf", "-march=rv64gcv"};
  if (target.rfind("amdgcn", 0) == 0) return {"--target=amdgcn-amd-amdhsa", "-mcpu=gfx1100"};
  if (target.rfind("nvptx", 0) == 0) return {"--target=nvptx64-nvidia-cuda", "-march=" + sm};
  return {};  // host: нативный clang без --target
}

bool has_suffix(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string stem_of(const std::string& path) {
  std::string b = path;
  const size_t slash = b.find_last_of("/\\");
  if (slash != std::string::npos) b = b.substr(slash + 1);
  const size_t dot = b.find_last_of('.');
  if (dot != std::string::npos) b = b.substr(0, dot);
  if (b.empty()) b = "a";
  return b;
}

// Есть ли исполняемый файл в PATH (без shell).
bool find_in_path(const std::string& prog, std::string* full = nullptr) {
  if (prog.find('/') != std::string::npos) return ::access(prog.c_str(), X_OK) == 0;
  const char* penv = ::getenv("PATH");
  const std::string path = penv ? penv : "/usr/bin:/bin";
  size_t i = 0;
  while (i <= path.size()) {
    size_t j = path.find(':', i);
    if (j == std::string::npos) j = path.size();
    const std::string cand = path.substr(i, j - i) + "/" + prog;
    if (::access(cand.c_str(), X_OK) == 0) {
      if (full) *full = cand;
      return true;
    }
    i = j + 1;
  }
  return false;
}

// Каталог текущего exe через /proc/self/exe (Linux), "" если недоступен.
std::string self_dir() {
  char buf[4096];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string p(buf);
  const size_t s = p.find_last_of('/');
  return (s == std::string::npos) ? "" : p.substr(0, s);
}

// Резолв ptx2ir: $ROUGE_PTX2IR, рядом с rouge-cc, затем PATH.
std::string resolve_ptx2ir() {
  if (const char* e = ::getenv("ROUGE_PTX2IR")) {
    if (*e) return e;
  }
  const std::string d = self_dir();
  if (!d.empty()) {
    const std::string cand = d + "/ptx2ir";
    if (::access(cand.c_str(), X_OK) == 0) return cand;
  }
  return "ptx2ir";  // надежда на PATH; execvp сам резолвит
}

std::string resolve_nvcc() {
  if (const char* e = ::getenv("ROUGE_NVCC")) {
    if (*e) return e;
  }
  return "nvcc";
}

std::string resolve_clang() {
  if (const char* e = ::getenv("ROUGE_CLANG")) {
    if (*e) return e;
  }
  return "clang";
}

// Только печать команды (для --dry-run): без fork/exec.
void print_cmd(const std::vector<std::string>& args) {
  std::string line;
  for (const auto& a : args) {
    if (!line.empty()) line += ' ';
    if (a.find_first_of(" \t'\"") == std::string::npos)
      line += a;
    else {
      line += '\'';
      for (char c : a) {
        if (c == '\'')
          line += "'\\''";
        else
          line += c;
      }
      line += '\'';
    }
  }
  std::fprintf(stderr, "+ %s\n", line.c_str());
}

// fork + execvp + waitpid. argv[0..]. Возвращает код выхода (0 = ok),
// либо 127 если запуск не удался. verbose печатает команду.
int run_cmd(const std::vector<std::string>& args, bool verbose) {
  if (verbose) print_cmd(args);

  std::vector<char*> argv;
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);

  pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "rouge-cc: fork failed\n");
    return 127;
  }
  if (pid == 0) {
    ::execvp(argv[0], argv.data());
    std::fprintf(stderr, "rouge-cc: cannot exec '%s': %s\n", argv[0], ::strerror(errno));
    ::_exit(127);
  }
  int st = 0;
  while (::waitpid(pid, &st, 0) < 0) {
  }
  if (WIFEXITED(st)) return WEXITSTATUS(st);
  return 127;
}

// Временный файл с суффиксом, возвращает путь (файл создан, пуст).
std::string make_temp(const std::string& suffix) {
  std::string tpl = "/tmp/rouge-cc-XXXXXX" + suffix;
  std::vector<char> buf(tpl.begin(), tpl.end());
  buf.push_back('\0');
  int fd = ::mkstemps(buf.data(), static_cast<int>(suffix.size()));
  if (fd < 0) return "";
  ::close(fd);
  return buf.data();
}

void usage(const char* prog) {
  std::printf(
      "usage: %s [options] <input.cu | input.ptx> [...]\n"
      "\n"
      "nvcc-compatible subset lowered through ptx2ir + clang.\n"
      "  -arch=sm_XX | -arch sm_XX      GPU arch for nvcc and nvptx backend\n"
      "  -arch=native                   host build (ignores SM arch)\n"
      "  --gpu-architecture=sm_XX        alias of -arch=\n"
      "  -ptx                           emit .ptx only (no object)\n"
      "  -c                             compile only (no link)\n"
      "  -o <file>                      output file\n"
      "  -I<dir> | -I <dir>              include path\n"
      "  -L<dir> | -L <dir>              library path (link step)\n"
      "  -D<mac>[=val] | -D <mac>        preprocessor define\n"
      "  -O0/-O1/-O2/-O3                 optimization level (clang step)\n"
      "  -g                             debug info (clang step)\n"
      "  --target <t> | --target=<t>     ROUGE-V extension: target platform\n"
      "                                 host | riscv | amd | nvptx\n"
      "                                 or full triple (x86_64-pc-linux-gnu,\n"
      "                                 riscv64-unknown-elf, amdgcn-amd-amdhsa,\n"
      "                                 nvptx64-nvidia-cuda). Default: host.\n"
      "  -v, --verbose                  print executed commands\n"
      "  -n, --dry-run                  print plan, execute nothing\n"
      "  -h, --help                     this help\n"
      "  --version                      version\n"
      "\n"
      ".cu inputs need a real `nvcc` in PATH (used as a black box for\n"
      ".cu -> .ptx only). .ptx inputs work without nvcc.\n",
      prog);
}

// Парсинг argv. Возвращает 0 = ok, иначе код выхода; err = текст ошибки.
int parse_args(int argc, char** argv, Options* o, std::string* err) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_val = [&](std::string* dst) -> bool {
      if (++i >= argc) {
        *err = "option '" + a + "' needs a value";
        return false;
      }
      *dst = argv[i];
      return true;
    };
    if (a == "-h" || a == "--help" || a == "-help") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--version" || a == "-version") {
      std::printf("rouge-cc 0.1.0 (ROUGE-V nvcc wrapper: nvcc -> ptx2ir -> clang)\n");
      std::exit(0);
    } else if (a == "-ptx" || a == "--ptx") {
      o->emitPtx = true;
    } else if (a == "-c" || a == "--compile") {
      o->compileOnly = true;
    } else if (a == "-o" || a == "--output-file") {
      if (!need_val(&o->output)) return 2;
    } else if (a == "-v" || a == "--verbose") {
      o->verbose = true;
    } else if (a == "-n" || a == "--dry-run") {
      o->dryRun = true;
      o->verbose = true;
    } else if (a == "-g" || a == "--debug") {
      o->debugInfo = true;
    } else if (a == "--target") {
      std::string v;
      if (!need_val(&v)) return 2;
      const std::string c = canonical_target(v);
      if (c.empty()) {
        *err = "unknown --target '" + v + "' (want host|riscv|amd|nvptx or full triple)";
        return 2;
      }
      o->target = c;
    } else if (a.rfind("--target=", 0) == 0) {
      const std::string v = a.substr(9);
      const std::string c = canonical_target(v);
      if (c.empty()) {
        *err = "unknown --target '" + v + "' (want host|riscv|amd|nvptx or full triple)";
        return 2;
      }
      o->target = c;
    } else if (a == "-arch" || a == "--gpu-architecture") {
      std::string v;
      if (!need_val(&v)) return 2;
      if (v == "native") { /* host build, SM не важен */
      } else if (v.rfind("sm_", 0) == 0) {
        o->smArch = v;
      } else {
        *err = "unsupported -arch '" + v + "' (want sm_XX or native)";
        return 2;
      }
    } else if (a.rfind("-arch=", 0) == 0 || a.rfind("--gpu-architecture=", 0) == 0) {
      const std::string v = a.substr(a.find('=') + 1);
      if (v == "native") {
      } else if (v.rfind("sm_", 0) == 0) {
        o->smArch = v;
      } else {
        *err = "unsupported -arch '" + v + "' (want sm_XX or native)";
        return 2;
      }
    } else if (a == "-I") {
      std::string v;
      if (!need_val(&v)) return 2;
      o->includes.push_back(v);
    } else if (a.rfind("-I", 0) == 0 && a.size() > 2) {
      o->includes.push_back(a.substr(2));
    } else if (a == "-L") {
      std::string v;
      if (!need_val(&v)) return 2;
      o->libdirs.push_back(v);
    } else if (a.rfind("-L", 0) == 0 && a.size() > 2) {
      o->libdirs.push_back(a.substr(2));
    } else if (a == "-D") {
      std::string v;
      if (!need_val(&v)) return 2;
      o->defines.push_back(v);
    } else if (a.rfind("-D", 0) == 0 && a.size() > 2) {
      o->defines.push_back(a.substr(2));
    } else if (a == "-O0" || a == "-O1" || a == "-O2" || a == "-O3" || a == "-O") {
      o->optLevel = (a == "-O") ? "-O2" : a;
    } else if (a == "-fatbin" || a == "-cubin" || a == "-dlink" || a == "--shared" || a == "-lib" ||
               a == "--lib") {
      *err = "option '" + a + "' is not supported by rouge-cc";
      return 2;
    } else if (!a.empty() && a[0] == '-') {
      *err = "unknown option '" + a + "'";
      return 2;
    } else {
      o->inputs.push_back(a);
    }
  }
  if (o->inputs.empty()) {
    *err = "no input files (.cu or .ptx expected)";
    return 2;
  }
  for (const auto& in : o->inputs) {
    if (has_suffix(in, ".cu") || has_suffix(in, ".ptx") || has_suffix(in, ".ll") ||
        has_suffix(in, ".o") || has_suffix(in, ".a"))
      continue;
    *err = "unsupported input '" + in + "' (want .cu, .ptx, .ll, .o, .a)";
    return 2;
  }
  if (!o->output.empty() && o->inputs.size() > 1 && (o->emitPtx || o->compileOnly)) {
    *err = "-o with multiple inputs requires a link step (drop -c/-ptx)";
    return 2;
  }
  // Device objects have no device linker here: linking works for host only.
  if (!o->emitPtx && !o->compileOnly && o->target != "x86_64-pc-linux-gnu") {
    *err = "linking is only supported for --target host (device targets: use -c or -ptx)";
    return 2;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  std::string err;
  if (int rc = parse_args(argc, argv, &o, &err)) {
    std::fprintf(stderr, "rouge-cc: %s\n", err.c_str());
    return rc;
  }

  const bool needNvcc =
      std::any_of(o.inputs.begin(), o.inputs.end(), [](const std::string& s) {
        return has_suffix(s, ".cu");
      });
  if (needNvcc && !find_in_path(resolve_nvcc())) {
    std::fprintf(stderr,
                 "rouge-cc: cannot compile '.cu' without a real nvcc in PATH.\n"
                 "  ROUGE-V has no CUDA-C frontend: install CUDA toolkit or pass\n"
                 "  a pre-generated .ptx file instead. Nothing was compiled.\n");
    if (!o.dryRun) return 1;
    // В dry-run продолжаем: показываем план (с пометкой), ничего не выполняя.
    std::fprintf(stderr, "rouge-cc: dry-run, showing plan anyway\n");
  }
  const std::string ptx2ir = resolve_ptx2ir();
  const std::string nvcc = resolve_nvcc();
  const std::string clang = resolve_clang();
  const std::vector<std::string> bflags = backend_flags(o.target, o.smArch);

  std::vector<std::string> objects;      // для шага линковки
  std::vector<std::string> temporaries;  // удалить перед выходом
  // В dry-run имена временные, но файлы не создаём (не сорим в /tmp).
  int dryN = 0;
  auto tmp = [&](const std::string& suf) -> std::string {
    if (o.dryRun) return "/tmp/rouge-cc-dryrun-" + std::to_string(dryN++) + suf;
    return make_temp(suf);
  };
  int rc = 0;

  for (const std::string& in : o.inputs) {
    std::string ptxPath = in;  // для .cu перезапишем путём из nvcc
    const bool isCu = has_suffix(in, ".cu");

    // --- Шаг 1: .cu -> .ptx через настоящий nvcc (чёрный ящик). ---
    if (isCu) {
      std::string outPtx;
      if (o.emitPtx) {
        outPtx = !o.output.empty() ? o.output : stem_of(in) + ".ptx";
      } else {
        outPtx = tmp(".ptx");
        if (outPtx.empty()) {
          std::fprintf(stderr, "rouge-cc: cannot create temp file\n");
          rc = 2;
          break;
        }
        temporaries.push_back(outPtx);
      }
      std::vector<std::string> cmd = {nvcc, "-ptx", "-arch=" + o.smArch, in, "-o", outPtx};
      for (const auto& d : o.includes) {
        cmd.push_back("-I");
        cmd.push_back(d);
      }
      for (const auto& d : o.defines) {
        cmd.push_back("-D");
        cmd.push_back(d);
      }
      if (o.dryRun) {
        print_cmd(cmd);
      } else {
        // Печать команды при -v — внутри run_cmd.
        rc = run_cmd(cmd, o.verbose);
        if (rc != 0) {
          std::fprintf(stderr, "rouge-cc: nvcc failed for '%s' (rc=%d)\n", in.c_str(), rc);
          break;
        }
      }
      if (o.emitPtx) {
        std::printf("%s\n", outPtx.c_str());
        continue;  // -ptx: дальше ничего не делаем
      }
      ptxPath = outPtx;
    } else if (o.emitPtx) {
      // -ptx с входом .ptx: нечего извлекать — копируем как есть.
      const std::string outPtx = !o.output.empty() ? o.output : stem_of(in) + ".ptx";
      if (o.dryRun) {
        std::fprintf(stderr, "+ cp '%s' '%s'\n", in.c_str(), outPtx.c_str());
      } else {
        std::vector<std::string> cmd = {"cp", in, outPtx};
        rc = run_cmd(cmd, o.verbose);
        if (rc != 0) break;
      }
      std::printf("%s\n", outPtx.c_str());
      continue;
    }

    // --- Шаг 2: .ptx -> .ll через ptx2ir. ---
    std::string llPath = in;
    if (has_suffix(ptxPath, ".ptx") || has_suffix(ptxPath, ".cu")) {
      llPath = tmp(".ll");
      if (llPath.empty()) {
        std::fprintf(stderr, "rouge-cc: cannot create temp file\n");
        rc = 2;
        break;
      }
      temporaries.push_back(llPath);
      const std::vector<std::string> cmd = {ptx2ir, "--target", o.target, ptxPath, llPath};
      if (o.dryRun) {
        print_cmd(cmd);
      } else {
        rc = run_cmd(cmd, o.verbose);
        if (rc != 0) {
          std::fprintf(stderr, "rouge-cc: ptx2ir failed for '%s' (rc=%d)\n", ptxPath.c_str(), rc);
          break;
        }
      }
    }
    // .ll вход используется напрямую (llPath = in).

    // --- Шаг 3: .ll -> .o через clang. ---
    std::string objPath;
    if (o.compileOnly) {
      objPath = !o.output.empty() ? o.output : stem_of(in) + ".o";
    } else {
      objPath = tmp(".o");
      if (objPath.empty()) {
        std::fprintf(stderr, "rouge-cc: cannot create temp file\n");
        rc = 2;
        break;
      }
      temporaries.push_back(objPath);
    }
    {
      std::vector<std::string> cmd = {clang};
      for (const auto& f : bflags) cmd.push_back(f);
      cmd.push_back(o.optLevel);
      if (o.debugInfo) cmd.push_back("-g");
      cmd.push_back("-c");
      cmd.push_back(llPath);
      cmd.push_back("-o");
      cmd.push_back(objPath);
      for (const auto& d : o.includes) {
        cmd.push_back("-I");
        cmd.push_back(d);
      }
      for (const auto& d : o.defines) {
        cmd.push_back("-D");
        cmd.push_back(d);
      }
      if (o.dryRun) {
        print_cmd(cmd);
      } else {
        rc = run_cmd(cmd, o.verbose);
        if (rc != 0) {
          std::fprintf(stderr, "rouge-cc: clang failed for '%s' (rc=%d)\n", llPath.c_str(), rc);
          break;
        }
      }
    }
    if (o.compileOnly) {
      std::printf("%s\n", objPath.c_str());
    } else {
      objects.push_back(objPath);
    }
  }

  // .o/.a входы сразу идут в линковку.
  if (rc == 0 && !o.compileOnly && !o.emitPtx) {
    for (const auto& in : o.inputs) {
      if (has_suffix(in, ".o") || has_suffix(in, ".a")) objects.push_back(in);
    }
  }

  // --- Шаг 4: линковка (если не -c и не -ptx). ---
  if (rc == 0 && !o.compileOnly && !o.emitPtx) {
    const std::string exe = !o.output.empty() ? o.output : "a.out";
    std::vector<std::string> cmd = {clang};
    for (const auto& f : bflags) cmd.push_back(f);
    cmd.push_back(o.optLevel);
    if (o.debugInfo) cmd.push_back("-g");
    for (const auto& ob : objects) cmd.push_back(ob);
    for (const auto& d : o.libdirs) {
      cmd.push_back("-L");
      cmd.push_back(d);
    }
    cmd.push_back("-o");
    cmd.push_back(exe);
    if (o.dryRun) {
      print_cmd(cmd);
    } else {
      rc = run_cmd(cmd, o.verbose);
      if (rc != 0)
        std::fprintf(stderr, "rouge-cc: link failed (rc=%d)\n", rc);
      else
        std::printf("%s\n", exe.c_str());
    }
  }

  for (const auto& t : temporaries) {
    if (!o.dryRun) ::remove(t.c_str());
  }
  return rc;
}
