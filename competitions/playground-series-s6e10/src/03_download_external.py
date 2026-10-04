"""M3 · 下载外部数据（比赛规则第 2.6 条允许）。

思路：
  比赛的 data-description 页明确写了本数据集"inspired by the Airline satisfaction
  dataset"，并给出了链接。规则 2.6.a 允许使用"公开且对所有参赛者同等可得"的外部数据。
  因此把这份*真实*数据加进训练是合规的，而且很可能是最大的分数提升来源。

  直觉：合成数据是从原始数据学分布后采样出来的，信息量上限不会超过原始数据。
  把原始数据一起训练，相当于在"被平滑过的数据"之外补回了真实的细节。
  但两者分布并不完全一致（官方原话："close to, but not exactly the same"），
  所以**不能盲目混入**——必须用 CV 验证，并可能要给它降权。

用法：
    python src/03_download_external.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "data" / "external"

DATASET = "arseniyshutko/binary-aviation-satisfaction-129k"


def main() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET, "-p", str(EXTERNAL_DIR), "-o",
    ]
    print("[下载] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    for zpath in sorted(EXTERNAL_DIR.glob("*.zip")):
        print(f"[解压] {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(EXTERNAL_DIR)

    print("\n[完成] data/external/ 内容：")
    for p in sorted(EXTERNAL_DIR.iterdir()):
        if p.is_file():
            print(f"  {p.name:40s} {p.stat().st_size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
