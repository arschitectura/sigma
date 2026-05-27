"""Shared palette utilities for tree visualizations."""

from __future__ import annotations

import colorsys

_LOW_HUE = 0.0
_MEDIUM_HUE = 50.0
_HIGH_HUE = 120.0


def _leaf_color(leaf_id: int, n_leaves: int) -> str:
    """Return the palette color for a leaf at the given 0-based position."""
    last_leaf_id = n_leaves - 1
    interpolation = leaf_id / max(last_leaf_id, 1)
    if interpolation <= 0.5:
        hue_deg = _LOW_HUE + (interpolation / 0.5) * (_MEDIUM_HUE - _LOW_HUE)
    else:
        hue_deg = _MEDIUM_HUE + ((interpolation - 0.5) / 0.5) * (
            _HIGH_HUE - _MEDIUM_HUE
        )
    red, green, blue = colorsys.hsv_to_rgb(hue_deg / 360.0, 0.9, 0.85)
    color = f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"
    return color


def _relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.0 relative luminance from a hex color string."""
    red = int(hex_color[1:3], 16) / 255.0
    green = int(hex_color[3:5], 16) / 255.0
    blue = int(hex_color[5:7], 16) / 255.0
    channels = []
    for channel in (red, green, blue):
        linear = (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
        channels.append(linear)
    luminance = (
        0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    )
    return luminance


def _contrast_foreground(hex_color: str) -> str:
    """Return "black" or "white" for best contrast per WCAG 2.0."""
    luminance = _relative_luminance(hex_color)
    foreground = "black" if luminance > 0.4 else "white"
    return foreground
