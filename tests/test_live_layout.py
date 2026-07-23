from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from src.capture.screen_geometry import (
    MacWindowCapture,
    ScreenGeometry,
    WindowAvailabilityError,
    WindowCaptureStatus,
    WindowServerInfo,
    parse_desktop_bounds,
    parse_window_server_candidates,
)
from src.pipeline.calibration import WindowInfo
from src.pipeline.live_layout import (
    LiveLayoutConfig,
    NormalizedBox,
    load_live_layout,
    render_layout_preview,
    render_roi_contact_sheet,
    save_live_layout,
)


def test_retina_geometry_maps_logical_window_to_pixels() -> None:
    geometry = ScreenGeometry(
        logical_size=(1470, 956),
        pixel_size=(2940, 1912),
    )

    assert geometry.scale_x == 2.0
    assert geometry.scale_y == 2.0
    assert geometry.logical_to_pixel_box((147, 82, 1323, 849)) == (
        294,
        164,
        2646,
        1698,
    )
    assert parse_desktop_bounds("0, 0, 1470, 956") == (1470, 956)


def test_mac_window_capture_uses_pixel_box() -> None:
    captured: dict[str, object] = {}

    def grabber(*, bbox):
        captured["bbox"] = bbox
        return Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]))

    source = MacWindowCapture(
        "斗地主",
        window_finder=lambda _: WindowInfo(
            app_name="斗地主",
            window_name="斗地主",
            window_box=(10, 20, 110, 120),
        ),
        geometry_provider=lambda: ScreenGeometry((200, 200), (400, 400)),
        grabber=grabber,
        prefer_window_level=False,
        clock=lambda: 123.0,
    )

    frame = source.capture(7)

    assert captured["bbox"] == (20, 40, 220, 240)
    assert frame.frame_id == 7
    assert frame.timestamp == 123.0
    assert frame.image.size == (200, 200)


def test_mac_window_capture_uses_window_server_image() -> None:
    window = WindowInfo(
        app_name="斗地主",
        window_name="斗地主",
        window_box=(10, 20, 110, 120),
    )
    source = MacWindowCapture(
        "斗地主",
        geometry_provider=lambda: ScreenGeometry((200, 200), (400, 400)),
        window_server_finder=lambda _: WindowServerInfo(42, window),
        window_grabber=lambda _: Image.new("RGB", (200, 200), "navy"),
        clock=lambda: 123.0,
    )

    frame = source.capture(7)

    assert frame.window == window
    assert frame.pixel_box == (20, 40, 220, 240)
    assert frame.image.getpixel((0, 0)) == (0, 0, 128)


def test_mac_window_capture_rejects_minimized_window() -> None:
    window = WindowInfo(
        app_name="斗地主",
        window_name="斗地主",
        window_box=(10, 20, 110, 120),
    )
    source = MacWindowCapture(
        "斗地主",
        geometry_provider=lambda: ScreenGeometry((200, 200), (400, 400)),
        window_server_finder=lambda _: WindowServerInfo(
            42,
            window,
            is_onscreen=False,
        ),
        window_grabber=lambda _: Image.new("RGB", (200, 200), "navy"),
    )

    with pytest.raises(WindowAvailabilityError) as error:
        source.capture(1)

    assert error.value.status is WindowCaptureStatus.MINIMIZED


def test_mac_window_capture_detects_minimize_during_capture() -> None:
    window = WindowInfo(
        app_name="斗地主",
        window_name="斗地主",
        window_box=(10, 20, 110, 120),
    )
    states = iter((
        WindowServerInfo(42, window, is_onscreen=True),
        WindowServerInfo(42, window, is_onscreen=False),
    ))
    source = MacWindowCapture(
        "斗地主",
        geometry_provider=lambda: ScreenGeometry((200, 200), (400, 400)),
        window_server_finder=lambda _: next(states),
        window_grabber=lambda _: Image.new("RGB", (180, 180), "navy"),
    )

    with pytest.raises(WindowAvailabilityError) as error:
        source.capture(1)

    assert error.value.status is WindowCaptureStatus.MINIMIZED


def test_parse_window_server_candidates() -> None:
    output = (
        "65272\t157.0\t86.0\t66.0\t20.0\t0\t0\tWindow\n"
        "65244\t147.0\t82.0\t1176.0\t767.0\t0\t1\t斗地主\n"
    )

    candidates = parse_window_server_candidates(output, app_name="斗地主")

    assert candidates[1] == (
        WindowServerInfo(
            window_id=65244,
            window=WindowInfo(
                app_name="斗地主",
                window_name="斗地主",
                window_box=(147, 82, 1323, 849),
            ),
            is_onscreen=True,
        ),
        0,
    )


def test_live_layout_roundtrip_and_preview(tmp_path: Path) -> None:
    config = LiveLayoutConfig(
        rois={
            **LiveLayoutConfig().rois,
            "self_turn": NormalizedBox(0.4, 0.4, 0.6, 0.6),
        },
        log_file=tmp_path / "live.jsonl",
    )
    path = tmp_path / "live.json"
    save_live_layout(path, config)

    loaded = load_live_layout(path)
    image = Image.new("RGB", (1000, 600), "navy")
    preview = render_layout_preview(image, loaded)
    contact = render_roi_contact_sheet(image, loaded, cell_size=(120, 80))

    assert loaded.roi("self_turn") == NormalizedBox(0.4, 0.4, 0.6, 0.6)
    assert preview.size == image.size
    assert contact.width == 360
    assert contact.height > 80
