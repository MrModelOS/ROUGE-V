#!/usr/bin/env python3
"""Извлекает struct cudaDeviceProp из настоящего driver_types.h NVIDIA.

Нужен для tests/test_device_prop_layout.cpp: тот сверяет раскладку нашей
rouge_cuda_device_prop с настоящей побайтово. Выделяем ТОЛЬКО структуру
(плюс cudaUUID_t) в отдельный заголовок, чтобы не тянуть весь настоящий
cuda_runtime_api.h в одну единицу трансляции с нашим (там конфликт
определений cudaError_t — оба заголовка объявляют одно и то же имя).

Использование:
  gen_real_prop_header.py <driver_types.h> <output.h>

Структура плоская (без вложенных фигурных скобок), это проверяется явно:
если NVIDIA изменит формат, скрипт упадёт, а не выдаст молча неверный файл.
"""
import re
import sys


def main() -> int:
    src_path, out_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8", errors="replace") as f:
        src = f.read()

    start = src.find("struct __device_builtin__ cudaDeviceProp")
    if start == -1:
        print("cudaDeviceProp struct not found", file=sys.stderr)
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

    # Комментарии вида /**< ... */ в одну строку — выкидываем построчно.
    lines = []
    for raw in body.split("\n"):
        code = raw.split("/**")[0].rstrip()
        if code.strip():
            lines.append(code)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "// Сгенерировано gen_real_prop_header.py из настоящего driver_types.h.\n"
            "// Только struct cudaDeviceProp (+ cudaUUID_t), без остального заголовка:\n"
            "// полный настоящий заголовок конфликтует с нашим в одной единице\n"
            "// трансляции (оба объявляют cudaError_t).\n"
            "#pragma once\n\n"
            "#include <stddef.h>\n\n"
            "typedef struct real_cuda_uuid_st {\n"
            "  char bytes[16];\n"
            "} real_cudaUUID_t;\n\n"
            "typedef struct real_cuda_device_prop_st {\n"
        )
        for line in lines:
            # cudaUUID_t внутри настоящей структуры — наш локальный алиас.
            line = line.replace("cudaUUID_t", "real_cudaUUID_t")
            f.write(line + "\n")
        # Имя типа оставляем настоящим: тест сравнивает именно cudaDeviceProp.
        f.write("} cudaDeviceProp;\n")
    print("wrote %s (%d fields)" % (out_path, len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
