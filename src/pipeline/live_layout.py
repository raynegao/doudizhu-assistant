"""Compatibility imports for the Phase 6 live layout configuration.

The configuration model lives in :mod:`src.config.live_layout` so the vision
layer does not depend on the pipeline orchestrator.  Keep this module as a
stable import path for local scripts and older integrations.
"""

from src.config.live_layout import (
    REQUIRED_LIVE_ROIS,
    LiveLayoutConfig,
    NormalizedBox,
    live_layout_from_dict,
    load_live_layout,
    render_layout_preview,
    render_roi_contact_sheet,
    save_live_layout,
)

__all__ = [
    "LiveLayoutConfig",
    "NormalizedBox",
    "REQUIRED_LIVE_ROIS",
    "live_layout_from_dict",
    "load_live_layout",
    "render_layout_preview",
    "render_roi_contact_sheet",
    "save_live_layout",
]
