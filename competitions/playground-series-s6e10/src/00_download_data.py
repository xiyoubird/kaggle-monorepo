"""M1 · 下载比赛数据。

思路：
- Kaggle 官方 CLI 会自动读取 ~/.kaggle/kaggle.json 里的凭据。
- 下载后解压到 data/raw/，之后所有脚本都从这里读，路径固定，
  保证任何一步都可以从零复现。

用法：
    python src/00_download_data.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "playground-series-s6e10"

# 以本文件位置为锚点定位项目根目录，避免依赖"当前工作目录"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # -c 指定竞赛，-p 指定下载目录，-o 允许覆盖已存在的文件
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", str(RAW_DIR),
        "-o",
    ]
    print("[下载] " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    # 解压所有 zip（Kaggle 有时会分片下载成多个 zip）
    for zpath in sorted(RAW_DIR.glob("*.zip")):
        print(f"[解压] {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(RAW_DIR)

    print("\n[完成] data/raw/ 内容：")
    for p in sorted(RAW_DIR.iterdir()):
        if p.is_file():
            print(f"  {p.name:45s} {p.stat().st_size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
