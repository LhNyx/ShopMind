"""评测回归对比：改动提示词 / 检索参数前后各跑一次，对比指标变化。

用法：uv run scripts/eval_compare.py data/reports/metrics_base.json data/reports/metrics_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("用法：eval_compare.py <old.json> <new.json>")
    old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    new = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    old_version = old["meta"].get("catalog_version")
    new_version = new["meta"].get("catalog_version")
    drift = "" if old_version == new_version else "  ⚠️ 目录变了，指标归因需谨慎"
    print(f"目录版本：{old_version} → {new_version}{drift}")
    print(f"{'指标':<22}{'旧':>10}{'新':>10}{'Δ':>10}")
    for key, old_value in old["metrics"].items():
        new_value = new["metrics"].get(key)
        delta = new_value - old_value
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"{key:<22}{old_value:>10.4f}{new_value:>10.4f}{arrow}{delta:>+9.4f}")

    old_bad = {c["id"] for c in old.get("badcases", [])}
    new_bad = {c["id"] for c in new.get("badcases", [])}
    fixed, introduced = old_bad - new_bad, new_bad - old_bad
    if fixed:
        print(f"\n修复的坏例：{sorted(fixed)}")
    if introduced:
        print(f"新引入的坏例：{sorted(introduced)}")


if __name__ == "__main__":
    main()
