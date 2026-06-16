"""Shared palette utilities for tree visualizations."""

from __future__ import annotations

_DEFAULT_LEAF_PALETTE = ("#D91616", "#D9B816", "#16D916")


def _leaf_color(
    leaf_id: int, n_leaves: int, palette: tuple[str, str, str]
) -> str:
    """Interpolate the per-leaf color from a (low, mid, high) palette tuple."""
    low_color, mid_color, high_color = palette
    last_leaf_id = n_leaves - 1
    fraction = leaf_id / max(last_leaf_id, 1)
    if fraction <= 0.5:
        local_fraction = fraction / 0.5
        color = _interpolate_oklab(low_color, mid_color, local_fraction)
    else:
        local_fraction = (fraction - 0.5) / 0.5
        color = _interpolate_oklab(mid_color, high_color, local_fraction)
    return color


def _contrast_foreground(hex_color: str) -> str:
    """Return "black" or "white" for best contrast per WCAG 2.0."""
    luminance = _relative_luminance(hex_color)
    foreground = "black" if luminance > 0.4 else "white"
    return foreground


def _relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.0 relative luminance from a hex color string."""
    red_linear, green_linear, blue_linear = _hex_to_linear_rgb(hex_color)
    luminance = (
        0.2126 * red_linear + 0.7152 * green_linear + 0.0722 * blue_linear
    )
    return luminance


def _perceptual_midpoint(hex1: str, hex2: str) -> str:
    """Return the OKLab midpoint of two hex colors as a hex string."""
    midpoint = _interpolate_oklab(hex1, hex2, 0.5)
    return midpoint


def _interpolate_oklab(hex1: str, hex2: str, fraction: float) -> str:
    """Linearly interpolate between two hex colors in the OKLab space."""
    L1, a1, b1 = _hex_to_oklab(hex1)
    L2, a2, b2 = _hex_to_oklab(hex2)
    L = L1 + fraction * (L2 - L1)
    a = a1 + fraction * (a2 - a1)
    b = b1 + fraction * (b2 - b1)
    hex_color = _oklab_to_hex(L, a, b)
    return hex_color


def _hex_to_oklab(hex_color: str) -> tuple[float, float, float]:
    """Convert a #RRGGBB string to OKLab (L, a, b)."""
    red_linear, green_linear, blue_linear = _hex_to_linear_rgb(hex_color)
    l_long = (
        0.4122214708 * red_linear
        + 0.5363325363 * green_linear
        + 0.0514459929 * blue_linear
    )
    m_medium = (
        0.2119034982 * red_linear
        + 0.6806995451 * green_linear
        + 0.1073969566 * blue_linear
    )
    s_short = (
        0.0883024619 * red_linear
        + 0.2817188376 * green_linear
        + 0.6299787005 * blue_linear
    )
    l_prime = _cube_root(l_long)
    m_prime = _cube_root(m_medium)
    s_prime = _cube_root(s_short)
    L = 0.2104542553 * l_prime + 0.7936177850 * m_prime - 0.0040720468 * s_prime
    a = 1.9779984951 * l_prime - 2.4285922050 * m_prime + 0.4505937099 * s_prime
    b = 0.0259040371 * l_prime + 0.7827717662 * m_prime - 0.8086757660 * s_prime
    oklab = (L, a, b)
    return oklab


def _oklab_to_hex(L: float, a: float, b: float) -> str:
    """Convert OKLab (L, a, b) to a #RRGGBB string, clamping to sRGB gamut."""
    l_prime = L + 0.3963377774 * a + 0.2158037573 * b
    m_prime = L - 0.1055613458 * a - 0.0638541728 * b
    s_prime = L - 0.0894841775 * a - 1.2914855480 * b
    l_long = l_prime * l_prime * l_prime
    m_medium = m_prime * m_prime * m_prime
    s_short = s_prime * s_prime * s_prime
    red_linear = (
        4.0767416621 * l_long - 3.3077115913 * m_medium + 0.2309699292 * s_short
    )
    green_linear = (
        -1.2684380046 * l_long
        + 2.6097574011 * m_medium
        - 0.3413193965 * s_short
    )
    blue_linear = (
        -0.0041960863 * l_long
        - 0.7034186147 * m_medium
        + 1.7076147010 * s_short
    )
    red = _linear_to_srgb(red_linear)
    green = _linear_to_srgb(green_linear)
    blue = _linear_to_srgb(blue_linear)
    hex_color = _srgb_to_hex(red, green, blue)
    return hex_color


def _hex_to_srgb(hex_color: str) -> tuple[float, float, float]:
    """Parse a #RRGGBB string into a (red, green, blue) tuple in [0, 1]."""
    red = int(hex_color[1:3], 16) / 255.0
    green = int(hex_color[3:5], 16) / 255.0
    blue = int(hex_color[5:7], 16) / 255.0
    srgb = (red, green, blue)
    return srgb


def _hex_to_linear_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a #RRGGBB string to linear-sRGB (red, green, blue) in [0, 1]."""
    red, green, blue = _hex_to_srgb(hex_color)
    red_linear = _srgb_to_linear(red)
    green_linear = _srgb_to_linear(green)
    blue_linear = _srgb_to_linear(blue)
    linear_rgb = (red_linear, green_linear, blue_linear)
    return linear_rgb


def _srgb_to_hex(red: float, green: float, blue: float) -> str:
    """Encode (red, green, blue) channels in [0, 1] as a #RRGGBB string."""
    red_clamped = min(1.0, max(0.0, red))
    green_clamped = min(1.0, max(0.0, green))
    blue_clamped = min(1.0, max(0.0, blue))
    red_byte = int(round(red_clamped * 255))
    green_byte = int(round(green_clamped * 255))
    blue_byte = int(round(blue_clamped * 255))
    hex_color = f"#{red_byte:02X}{green_byte:02X}{blue_byte:02X}"
    return hex_color


def _srgb_to_linear(channel: float) -> float:
    """sRGB to linear-sRGB single-channel gamma decode (IEC 61966-2-1)."""
    if channel <= 0.04045:
        linear = channel / 12.92
    else:
        linear = ((channel + 0.055) / 1.055) ** 2.4
    return linear


def _linear_to_srgb(channel: float) -> float:
    """Linear-sRGB to sRGB single-channel gamma encode (IEC 61966-2-1)."""
    if channel <= 0.0031308:
        encoded = channel * 12.92
    else:
        encoded = 1.055 * (channel ** (1.0 / 2.4)) - 0.055
    return encoded


def _cube_root(value: float) -> float:
    """Real-valued cube root that preserves the sign of the input."""
    if value >= 0.0:
        result = value ** (1.0 / 3.0)
    else:
        result = -((-value) ** (1.0 / 3.0))
    return result
