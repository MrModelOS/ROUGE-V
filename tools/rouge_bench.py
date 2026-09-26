#!/usr/bin/env python3
"""
ROUGE-bench v0.1 — калькулятор $/токен.

Формула из docs/04-business-model.md:34:
    $/токен = (стоимость железа/срок службы + энергия + память + операции) / токенов за срок

    total_cost = hardware_cost + energy_cost + memory_cost + ops_cost
    tokens_lifetime = tokens_per_sec * 3600 * lifetime_hours
    cost_per_token = total_cost / tokens_lifetime

Сравнение: ROUGE конфигурация B vs NVIDIA конфигурация A (заглушки по умолчанию).
    NVIDIA конфигурация A : $30k, 350W
    ROUGE конфигурация B: $15k, 200W

Зависимости: только stdlib.

Примеры вызова:
    # 1. Справка
    python3 tools/rouge_bench.py --help

    # 2. Дефолтное сравнение (встроенные заглушки)
    python3 tools/rouge_bench.py

    # 3. Из JSON-файла
    python3 tools/rouge_bench.py tools/rouge_bench_example.json
    cat tools/rouge_bench_example.json | python3 tools/rouge_bench.py --stdin

    # 4. Переопределение полей через CLI
    python3 tools/rouge_bench.py --b-tps 2400 --h100-tps 2200 --energy-price 0.12

    # 5. Сохранить отчёт JSON
    python3 tools/rouge_bench.py --json-out report.json

Формат JSON (все поля опциональны, см. tools/rouge_bench_example.json):
    {
      "lifetime_hours": 26280,
      "energy_price": 0.12,
      "memory_cost": 1500,
      "ops_cost": 500,
      "h100": {"hardware_cost": 30000, "power_w": 350, "tokens_per_sec": 2200},
      "config_b": {"hardware_cost": 9000, "power_w": 280, "tokens_per_sec": 3100}
    }
    Плоский формат тоже поддерживается:
    {"hardware_cost": 9000, "power_w": 280, "tokens_per_sec": 3100, ...}

ВАЖНО: это КАЛЬКУЛЯТОР МЕТОДИКИ, а не измерение.
Скрипт считает $/токен по опубликованной формуле, подставляя переданные
вами входные данные. Он ничего не меряет на реальном железе: все числа
в примере условные и существуют только чтобы показать работу формулы.
Не публикуйте вывод этого скрипта как «результат теста» и не ссылайтесь
на него как на измеренное преимущество — без стенда это недостоверная
реклама. Настоящее сравнение требует замеров на своём железе при
одинаковых модели, батче и условиях у обеих сторон.
"""

import argparse
import json
import sys
import os

DEFAULTS = {
    "lifetime_hours": 26280,  # 3 года 24/7
    "energy_price": 0.12,     # $/кВт·ч
    "memory_cost": 0.0,       # $ за срок (HBM/аренда памяти)
    "ops_cost": 0.0,          # $ за операции/обслуживание за срок
    "h100": {
        "name": "NVIDIA конфигурация A",
        "hardware_cost": 30000,
        "power_w": 350,
        "tokens_per_sec": 2200,
    },
    "config_b": {
        "name": "ROUGE конфигурация B",
        "hardware_cost": 15000,
        "power_w": 200,
        "tokens_per_sec": 3200,
    },
}

def compute(entry, lifetime_hours, energy_price, memory_cost, ops_cost):
    hw = float(entry["hardware_cost"])
    pw = float(entry["power_w"])
    tps = float(entry["tokens_per_sec"])
    # энергия за весь срок
    energy_kwh = pw * lifetime_hours / 1000.0
    energy_cost = energy_kwh * energy_price
    total_cost = hw + energy_cost + memory_cost + ops_cost
    tokens_lifetime = tps * 3600.0 * lifetime_hours
    cost_per_token = total_cost / tokens_lifetime if tokens_lifetime else float("inf")
    cost_per_1m = cost_per_token * 1_000_000
    cost_per_1k = cost_per_token * 1000
    # $/час амортизация железа
    hw_per_hour = hw / lifetime_hours if lifetime_hours else 0
    return {
        "name": entry.get("name", "unknown"),
        "hardware_cost": hw,
        "power_w": pw,
        "tokens_per_sec": tps,
        "energy_kwh": energy_kwh,
        "energy_cost": energy_cost,
        "memory_cost": memory_cost,
        "ops_cost": ops_cost,
        "total_cost": total_cost,
        "tokens_lifetime": tokens_lifetime,
        "cost_per_token": cost_per_token,
        "cost_per_1k": cost_per_1k,
        "cost_per_1m": cost_per_1m,
        "hw_per_hour": hw_per_hour,
    }

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_config(cli_args, file_cfg):
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if not file_cfg:
        file_cfg = {}

    # глобальные поля
    for k in ("lifetime_hours", "energy_price", "memory_cost", "ops_cost", "operations_cost"):
        if k in file_cfg:
            # operations_cost -> ops_cost алиас
            target = "ops_cost" if k == "operations_cost" else k
            cfg[target] = file_cfg[k]

    # поддержка плоского формата: если в корне есть hardware_cost/power_w/tokens_per_sec — это config_b
    flat_keys = {"hardware_cost", "power_w", "tokens_per_sec"}
    if flat_keys & set(file_cfg.keys()):
        for k in flat_keys:
            if k in file_cfg:
                cfg["config_b"][k] = file_cfg[k]
        # также плоские lifetime/energy уже скопированы выше
        if "name" in file_cfg:
            cfg["config_b"]["name"] = file_cfg["name"]

    # вложенные конфигурации h100 / config_b
    for dev in ("h100", "config_b"):
        if dev in file_cfg and isinstance(file_cfg[dev], dict):
            for k, v in file_cfg[dev].items():
                cfg[dev][k] = v
        # алиасы: nvidia -> h100, rouge -> config_b
        for alias, target in (("nvidia", "h100"), ("config_b", "config_b")):
            if alias in file_cfg and isinstance(file_cfg[alias], dict):
                for k, v in file_cfg[alias].items():
                    cfg[target][k] = v

    # CLI переопределения (высший приоритет)
    if cli_args.lifetime_hours is not None:
        cfg["lifetime_hours"] = cli_args.lifetime_hours
    if cli_args.energy_price is not None:
        cfg["energy_price"] = cli_args.energy_price
    if cli_args.memory_cost is not None:
        cfg["memory_cost"] = cli_args.memory_cost
    if cli_args.ops_cost is not None:
        cfg["ops_cost"] = cli_args.ops_cost
    if cli_args.h100_cost is not None:
        cfg["h100"]["hardware_cost"] = cli_args.h100_cost
    if cli_args.h100_power is not None:
        cfg["h100"]["power_w"] = cli_args.h100_power
    if cli_args.h100_tps is not None:
        cfg["h100"]["tokens_per_sec"] = cli_args.h100_tps
    if cli_args.b_cost is not None:
        cfg["config_b"]["hardware_cost"] = cli_args.b_cost
    if cli_args.b_power is not None:
        cfg["config_b"]["power_w"] = cli_args.b_power
    if cli_args.b_tps is not None:
        cfg["config_b"]["tokens_per_sec"] = cli_args.b_tps

    return cfg

def fmt_money(x):
    if x >= 1000:
        return f"${x:,.2f}"
    return f"${x:.4f}"

def fmt_num(x):
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    return f"{x:,.0f}"

def print_table(r_h100, r_config_b, lifetime_hours, energy_price):
    # заголовок
    print("=" * 78)
    print("$/токен — калькулятор методики  (формула: (железо + энергия + память + операции)/токены)")
    print(f"Срок службы: {lifetime_hours:,} ч ({lifetime_hours/8760:.1f} лет)  |  Энергия: ${energy_price}/кВт·ч")
    print("=" * 78)
    # таблица
    headers = ["", r_h100["name"], r_config_b["name"]]
    rows = [
        ["Железо, $", f"{r_h100['hardware_cost']:,.0f}", f"{r_config_b['hardware_cost']:,.0f}"],
        ["Мощность, Вт", f"{r_h100['power_w']:.0f}", f"{r_config_b['power_w']:.0f}"],
        ["Энергия за срок, кВт·ч", f"{r_h100['energy_kwh']:,.0f}", f"{r_config_b['energy_kwh']:,.0f}"],
        ["Стоимость энергии, $", f"{r_h100['energy_cost']:,.2f}", f"{r_config_b['energy_cost']:,.2f}"],
        ["Память+операции, $", f"{r_h100['memory_cost']+r_h100['ops_cost']:,.2f}", f"{r_config_b['memory_cost']+r_config_b['ops_cost']:,.2f}"],
        ["Итого затрат, $", f"{r_h100['total_cost']:,.2f}", f"{r_config_b['total_cost']:,.2f}"],
        ["Токенов/сек", f"{r_h100['tokens_per_sec']:,.0f}", f"{r_config_b['tokens_per_sec']:,.0f}"],
        ["Токенов за срок", fmt_num(r_h100['tokens_lifetime']), fmt_num(r_config_b['tokens_lifetime'])],
        ["$/токен", f"{r_h100['cost_per_token']:.8f}", f"{r_config_b['cost_per_token']:.8f}"],
        ["$/1K токенов", f"{r_h100['cost_per_1k']:.6f}", f"{r_config_b['cost_per_1k']:.6f}"],
        ["$/1M токенов", f"${r_h100['cost_per_1m']:.4f}", f"${r_config_b['cost_per_1m']:.4f}"],
    ]
    # ширины
    col_w = [22, 22, 22]
    def fmt_row(cells):
        return f"  {cells[0]:<{col_w[0]}} | {cells[1]:>{col_w[1]}} | {cells[2]:>{col_w[2]}}"
    print(fmt_row(headers))
    print("-" * 78)
    for r in rows:
        print(fmt_row(r))
    print("-" * 78)

    # Сравнение — это арифметика над ПЕРЕДАННЫМИ входными данными, а не
    # измерение. Формулировки намеренно нейтральные: скрипт не имеет права
    # объявлять «достижение», потому что ничего не измерял.
    ratio = r_h100["cost_per_token"] / r_config_b["cost_per_token"] if r_config_b["cost_per_token"] else 0
    print()
    print("ВНИМАНИЕ: это расчёт по входным данным из конфига, а не замер.")
    print("Для публикуемого сравнения нужны реальные замеры на стенде.")
    print()
    if ratio > 1:
        print(f"По заданным числам: конфигурация B дешевле в {ratio:.2f} раз "
              f"($/токен ниже на {(1-1/ratio)*100:.1f}%).")
    elif ratio < 1:
        inv = 1/ratio if ratio else 0
        print(f"По заданным числам: конфигурация A дешевле в {inv:.2f} раз.")
    else:
        print("По заданным числам: паритет.")

    # экономия
    saved = r_h100["total_cost"] - r_config_b["total_cost"]
    if saved > 0:
        print(f"  Разница в стоимости владения за срок, по заданным числам: ${saved:,.2f}")
    print()

def main():
    p = argparse.ArgumentParser(
        description="ROUGE-bench v0.1 — калькулятор $/токен (ROUGE конфигурация B vs NVIDIA конфигурация A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пример: python3 tools/rouge_bench.py tools/rouge_bench_example.json"
    )
    p.add_argument("input", nargs="?", help="JSON-файл с параметрами (опционально, иначе дефолты)")
    p.add_argument("--stdin", action="store_true", help="читать JSON из stdin")
    p.add_argument("--json-out", metavar="FILE", help="сохранить отчёт в JSON")
    p.add_argument("--lifetime-hours", type=float, help="срок службы, ч")
    p.add_argument("--energy-price", type=float, help="цена энергии $/кВт·ч")
    p.add_argument("--memory-cost", type=float, help="стоимость памяти за срок, $")
    p.add_argument("--ops-cost", type=float, help="стоимость операций за срок, $")
    p.add_argument("--h100-cost", type=float, help="цена конфигурация A, $")
    p.add_argument("--h100-power", type=float, help="мощность конфигурация A, Вт")
    p.add_argument("--h100-tps", type=float, help="токенов/сек конфигурация A")
    p.add_argument("--b-cost", type=float, help="цена конфигурация B, $")
    p.add_argument("--b-power", type=float, help="мощность конфигурация B, Вт")
    p.add_argument("--b-tps", type=float, help="токенов/сек конфигурация B")
    args = p.parse_args()

    file_cfg = None
    if args.stdin:
        file_cfg = json.load(sys.stdin)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Файл не найден: {args.input}", file=sys.stderr)
            sys.exit(2)
        file_cfg = load_json_file(args.input)

    cfg = merge_config(args, file_cfg)

    r_h100 = compute(cfg["h100"], cfg["lifetime_hours"], cfg["energy_price"], cfg["memory_cost"], cfg["ops_cost"])
    r_config_b = compute(cfg["config_b"], cfg["lifetime_hours"], cfg["energy_price"], cfg["memory_cost"], cfg["ops_cost"])

    print_table(r_h100, r_config_b, cfg["lifetime_hours"], cfg["energy_price"])

    if args.json_out:
        out = {
            "version": "0.1",
            "formula": "(hardware_cost + energy_cost + memory_cost + ops_cost) / tokens_lifetime",
            "lifetime_hours": cfg["lifetime_hours"],
            "energy_price": cfg["energy_price"],
            "memory_cost": cfg["memory_cost"],
            "ops_cost": cfg["ops_cost"],
            "h100": r_h100,
            "config_b": r_config_b,
            "ratio_h100_per_config_b": r_h100["cost_per_token"] / r_config_b["cost_per_token"] if r_config_b["cost_per_token"] else None,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Отчёт записан: {args.json_out}")

if __name__ == "__main__":
    main()
