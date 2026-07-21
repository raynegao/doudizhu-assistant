"""Create a publishable local GIF from the reproducible Phase 5B API evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.ui.web import DEFAULT_SCENARIOS, ShowcaseWebApp


def build_demo_gif(output: Path) -> Path:
    app = ShowcaseWebApp()
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    for scenario in DEFAULT_SCENARIOS:
        payload = app._decision(scenario)
        recommendation = payload["recommendation"]
        top = recommendation["top_k"][0]
        lines = ["Dou Dizhu Assistant | Phase 5B", f"Scenario: {scenario}", f"Recommended: {' '.join(recommendation['recommended_action'])}", f"Score: {top['strategy_score']:.3f}", f"Estimated win: {top['estimated_win_rate']:.1%}", "Read-only replay · no auto-play"]
        image = Image.new("RGB", (900, 360), "#101828")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((48, 42 + index * 48), line, font=font, fill="#f9fafb" if index < 2 else "#a6f4c5")
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1600, loop=0)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the local Phase 5B demonstration GIF.")
    parser.add_argument("--output", default="runs/phase5b/demo.gif")
    args = parser.parse_args(argv)
    print(build_demo_gif(Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
