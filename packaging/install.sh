#!/bin/sh
# ROUGE-V quick install: ptx2ir + rouge-run into ~/.local/bin.
# The binaries only need base system libraries (libstdc++/libc); nothing is
# put into ~/.local/lib because there is nothing to put there.
# You still need separately: clang (the backend, e.g. via your package
# manager) and optionally nvcc (to produce PTX from .cu sources).
set -eu

VER="${1:-0.1}"
PREFIX="${PREFIX:-$HOME/.local}"
REPO='https://github.com/MrModelOS/ROUGE-V'
TGZ="rouge-v-$VER-linux-x86_64.tar.gz"

command -v curl >/dev/null || { echo "install: need curl" >&2; exit 1; }
command -v tar >/dev/null || { echo "install: need tar" >&2; exit 1; }

mkdir -p "$PREFIX"
echo "install: $REPO/releases/download/v$VER/$TGZ -> $PREFIX"
curl -fsSL "$REPO/releases/download/v$VER/$TGZ" | tar xz -C "$PREFIX"

for t in ptx2ir rouge-run; do
  [ -x "$PREFIX/bin/$t" ] || { echo "install: $t missing after unpack" >&2; exit 1; }
done
echo "install: ptx2ir + rouge-run ready in $PREFIX/bin"
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "install: add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac
command -v clang >/dev/null || echo "install: note: clang not found (needed for -o / backend step)"
"$PREFIX/bin/ptx2ir" 2>&1 | head -1 || true
