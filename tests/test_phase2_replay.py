from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts import replay_phase2


def test_replay_phase2_connects_mock_predictions_to_rule_engine(tmp_path: Path, monkeypatch, capsys) -> None:
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    for index in range(16):
        Image.new("RGB", (20, 30), color=(255, 255, 255)).save(crop_dir / f"card_{index:02d}.png")

    ranks = ["A", "K", "Q", "J", "10", "10", "9", "8", "7", "7", "6", "5", "4", "3", "3", "BJ"]

    def fake_predict_crop_dir(model_path: Path, crop_dir: Path, device_name: str = "auto") -> list[dict[str, object]]:
        return [
            {"index": index, "file": str(crop_dir / f"card_{index:02d}.png"), "rank": rank, "confidence": 0.99}
            for index, rank in enumerate(ranks)
        ]

    monkeypatch.setattr(replay_phase2, "predict_crop_dir", fake_predict_crop_dir)

    assert replay_phase2.replay_phase2(
        model_path=tmp_path / "mock.pt",
        roi_path=None,
        screenshot_path=None,
        crop_dir=crop_dir,
        last_play="5 5",
        device_name="cpu",
        roi_box=replay_phase2.WINDOW_MODE_ROI_BOX,
        count=16,
        start_x=0,
        start_y=20,
        step_x=135,
        crop_size=(126, 210),
    ) == 0
    output = capsys.readouterr().out
    assert "识别手牌:" in output
    assert "候选动作数:" in output
    assert "推荐动作:" in output
