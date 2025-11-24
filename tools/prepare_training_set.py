"""
Merge multiple image/label sources into a single YOLO dataset folder.

Example (合并真实 + 合成 + 伪标签三套数据到 data/yolo_dataset_merged):
    python tools/prepare_training_set.py ^
        --image-dirs data/raw_screenshots,data/raw_screenshots_synth,data/raw_screenshots ^
        --label-dirs data/labels_yolo,data/labels_yolo_synth,data/labels_yolo_pseudo ^
        --out data/yolo_dataset_merged

注意：image-dirs 与 label-dirs 顺序一一对应，文件名需匹配（同名 png/txt）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple YOLO label sources into one dataset.")
    p.add_argument("--image-dirs", type=str, required=True, help="Comma-separated image dirs.")
    p.add_argument("--label-dirs", type=str, required=True, help="Comma-separated label dirs.")
    p.add_argument("--out", type=Path, default=Path("data/yolo_dataset_merged"), help="Output dataset root.")
    return p.parse_args()


def collect_pairs(img_dir: Path, lbl_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def stage_dataset(out_root: Path, pairs: List[Tuple[Path, Path]]) -> None:
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    if out_root.exists():
        shutil.rmtree(out_root)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)


def main() -> None:
    args = parse_args()
    img_dirs = [Path(p) for p in args.image_dirs.split(",") if p]
    lbl_dirs = [Path(p) for p in args.label_dirs.split(",") if p]
    if len(img_dirs) != len(lbl_dirs):
        raise ValueError("image-dirs 和 label-dirs 数量必须一致")

    all_pairs: List[Tuple[Path, Path]] = []
    for img_dir, lbl_dir in zip(img_dirs, lbl_dirs):
        pairs = collect_pairs(img_dir, lbl_dir)
        all_pairs.extend(pairs)
        print(f"[+] {img_dir} + {lbl_dir}: {len(pairs)} 对")

    if not all_pairs:
        raise RuntimeError("未找到任何匹配的图片+标签对，请检查路径与文件名是否对应。")

    stage_dataset(args.out, all_pairs)
    print(f"合并完成，共 {len(all_pairs)} 对，已写入 {args.out}")


if __name__ == "__main__":
    main()
