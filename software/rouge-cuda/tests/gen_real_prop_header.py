#!/usr/bin/env python3
import re
import sys
"""Извлекает структуру из настоящего driver_types.h NVIDIA.

Нужен для layout-тестов: наши структуры — ABI-контракт, поэтому их раскладка
сверяется с настоящей побайтово. Выделяем ТОЛЬКО структуру в отдельный
заголовок, чтобы не тянуть весь настоящий заголовок в одну единицу
трансляции с нашим (там конфликт определений cudaError_t).

Использование:
  gen_real_prop_header.py <driver_types.h> <StructName> <output.h> [typedef]

Без typedef тип называется как в настоящем заголовке. Структура обязана быть
плоской (без вложенных фигурных скобок) — это проверяется явно: если NVIDIA
изменит формат, скрипт упадёт, а не выдаст молча неверный файл.
"""

import re
import sys
import sys


def main() -> int:
    src_path, struct_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    typedef_name = sys.argv[4] if len(sys.argv) > 4 else struct_name

    with open(src_path, encoding="utf-8", errors="replace") as f:
        src = f.read()

    start = src.find("struct __device_builtin__ " + struct_name)
    if start == -1:
        print(struct_name + " struct not found", file=sys.stderr)
        return 1
    brace = src.find("{", start)
    depth = 0
    end = -1
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        print("unbalanced braces", file=sys.stderr)
        return 1
    body = src[brace + 1 : end]
    if "{" in body or "}" in body:
        print("nested braces: struct is not flat, refusing", file=sys.stderr)
        return 1

    # Все блочные комментарии /* ... */ выкидываем целиком (включая
    # многострочные вида /** ... */) — в структуре остаются только поля.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    lines = []
    for raw in body.split("\n"):
        code = raw.rstrip()
        if code.strip():
            lines.append(code)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Сгенерировано gen_real_prop_header.py из настоящего driver_types.h.\n")
        f.write("// Только struct " + struct_name + ", без остального заголовка:\n")
        f.write("// полный настоящий заголовок конфликтует с нашим в одной единице\n")
        f.write("// трансляции (оба объявляют cudaError_t).\n")
        f.write("#pragma once\n\n#include <stddef.h>\n\n")
        f.write("typedef struct real_cuda_uuid_st {\n  char bytes[16];\n} real_cudaUUID_t;\n\n")
        f.write("typedef struct real_cuda_prop_st {\n")
        for line in lines:
            # cudaUUID_t внутри настоящей структуры — наш локальный алиас.
            # cudaSharedMemoryMode — наш локальный enum (значения из того же заголовка).
            line = line.replace("cudaUUID_t", "real_cudaUUID_t")
            line = line.replace("cudaSharedMemoryMode", "rouge_cudaSharedMemoryMode")
            f.write(line + "\n")
        f.write("} " + typedef_name + ";\n")
    print("wrote %s (%d fields)" % (out_path, len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
