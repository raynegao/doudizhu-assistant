from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.add_labeled_crops_to_dataset import add_labeled_crops
from scripts.crop_hand_roi_cards import crop_hand_roi_cards
from scripts.generate_card_cls_dataset import generate_dataset
from scripts.prepare_card_templates import classify_template_name, collect_local_templates
from scripts.rebuild_card_cls_dataset import CropSource, rebuild_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 36), color=(240, 240, 240)).save(path)


def test_template_name_mapping_matches_douzero_conventions() -> None:
    assert classify_template_name("mbT") == "10"
    assert classify_template_name("mrD") == "BJ"
    assert classify_template_name("obX") == "SJ"
    assert classify_template_name("landlord") is None


def test_collect_local_templates_and_generate_dataset(tmp_path: Path) -> None:
    pics_dir = tmp_path / "pics"
    _write_png(pics_dir / "mb3.png")
    _write_png(pics_dir / "mrD.png")
    _write_png(pics_dir / "button.png")

    seed_dir = tmp_path / "seed"
    assert collect_local_templates(pics_dir, seed_dir) == 2
    assert (seed_dir / "3" / "local__mb3.png").exists()
    assert (seed_dir / "BJ" / "local__mrD.png").exists()

    output_dir = tmp_path / "cards_cls"
    generate_dataset(
        seed_dir=seed_dir,
        output_dir=output_dir,
        per_seed=2,
        val_ratio=0.5,
        image_size=(32, 48),
        seed=1,
    )

    assert len(list((output_dir / "train" / "3").glob("*.png"))) == 1
    assert len(list((output_dir / "val" / "3").glob("*.png"))) == 1
    assert len(list((output_dir / "train" / "BJ").glob("*.png"))) == 1
    assert len(list((output_dir / "val" / "BJ").glob("*.png"))) == 1


def test_crop_hand_roi_cards_uses_fixed_overlap_step(tmp_path: Path) -> None:
    roi = Image.new("RGB", (40, 12), color=(255, 255, 255))
    pixels = roi.load()
    for x, color in [(0, (255, 0, 0)), (10, (0, 255, 0)), (20, (0, 0, 255))]:
        for yy in range(12):
            for xx in range(x, x + 4):
                pixels[xx, yy] = color

    roi_path = tmp_path / "hand_roi.png"
    roi.save(roi_path)

    output_dir = tmp_path / "cards"
    metadata = crop_hand_roi_cards(
        roi_path=roi_path,
        output_dir=output_dir,
        count=3,
        start_x=0,
        start_y=0,
        step_x=10,
        crop_size=(4, 5),
    )

    assert [item["box"] for item in metadata] == [[0, 0, 4, 5], [10, 0, 14, 5], [20, 0, 24, 5]]
    assert Image.open(output_dir / "card_00.png").getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(output_dir / "card_01.png").getpixel((0, 0)) == (0, 255, 0)
    assert Image.open(output_dir / "card_02.png").getpixel((0, 0)) == (0, 0, 255)
    assert (output_dir / "metadata.json").exists()


def test_add_labeled_crops_to_dataset(tmp_path: Path) -> None:
    crop_dir = tmp_path / "crops"
    _write_png(crop_dir / "card_00.png")
    _write_png(crop_dir / "card_01.png")

    output_dir = tmp_path / "cards_cls"
    count = add_labeled_crops(
        crop_dir=crop_dir,
        labels=["A", "BJ"],
        output_dir=output_dir,
        per_crop=2,
        val_ratio=0.5,
        image_size=(32, 48),
        seed=1,
    )

    assert count == 4
    assert len(list((output_dir / "train" / "A").glob("*.png"))) + len(list((output_dir / "val" / "A").glob("*.png"))) == 2
    assert len(list((output_dir / "train" / "BJ").glob("*.png"))) + len(list((output_dir / "val" / "BJ").glob("*.png"))) == 2


def test_rebuild_dataset_writes_test_split_and_manifest(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_png(seed_dir / "3" / "template_3.png")

    crop_dir = tmp_path / "roi_samples" / "good_roi"
    _write_png(crop_dir / "card_00.png")
    _write_png(crop_dir / "card_01.png")

    bad_crop_dir = tmp_path / "roi_samples" / "hand_roi_001_step135"
    _write_png(bad_crop_dir / "card_00.png")

    output_dir = tmp_path / "cards_cls"
    summary = rebuild_dataset(
        seed_dir=seed_dir,
        output_dir=output_dir,
        crop_sources=(CropSource(crop_dir=crop_dir, labels=("A", "BJ")),),
        template_per_seed=5,
        real_per_crop=10,
        image_size=(32, 48),
        seed=1,
        clean=True,
    )

    assert summary["sample_count"] == 25
    assert (output_dir / "train" / "3").exists()
    assert (output_dir / "val" / "3").exists()
    assert list((output_dir / "test" / "A").glob("*.png"))
    assert list((output_dir / "test" / "BJ").glob("*.png"))

    manifest = (output_dir / "manifest.jsonl").read_text(encoding="utf-8")
    assert "good_roi" in manifest
    assert "hand_roi_001_step135" not in manifest
    assert "source_dir" in manifest
    assert "augmentation_seed" in manifest
