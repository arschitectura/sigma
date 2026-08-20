"""Tests for sigma._palette."""

from __future__ import annotations

import unittest

import sigma._palette


class TestInterpolateOklab(unittest.TestCase):
    """Tests for the OKLab-space color interpolation helper."""

    __slots__ = ()

    def test_endpoint_at_zero_recovers_first_color(self):
        """Interpolation at fraction 0 round-trips to (approximately) hex1."""
        result = sigma._palette._interpolate_oklab("#FF0000", "#00FF00", 0.0)
        self.assertEqual(result.upper(), "#FF0000")

    def test_endpoint_at_one_recovers_second_color(self):
        """Interpolation at fraction 1 round-trips to (approximately) hex2."""
        result = sigma._palette._interpolate_oklab("#FF0000", "#00FF00", 1.0)
        self.assertEqual(result.upper(), "#00FF00")

    def test_midpoint_of_identical_colors_is_that_color(self):
        """Interpolating a color with itself yields the same color at every t."""
        result = sigma._palette._interpolate_oklab("#123456", "#123456", 0.5)
        self.assertEqual(result.upper(), "#123456")


class TestPerceptualMidpoint(unittest.TestCase):
    """Tests for the OKLab perceptual midpoint helper."""

    __slots__ = ()

    def test_midpoint_of_black_and_white_is_a_neutral_gray(self):
        """The OKLab midpoint of black and white has equal R, G, B channels."""
        midpoint = sigma._palette._perceptual_midpoint("#000000", "#FFFFFF")
        red = int(midpoint[1:3], 16)
        green = int(midpoint[3:5], 16)
        blue = int(midpoint[5:7], 16)
        self.assertEqual(red, green)
        self.assertEqual(green, blue)

    def test_midpoint_is_symmetric_in_its_arguments(self):
        """Midpoint(a, b) and midpoint(b, a) produce the same hex color."""
        forward = sigma._palette._perceptual_midpoint("#112233", "#AABBCC")
        backward = sigma._palette._perceptual_midpoint("#AABBCC", "#112233")
        self.assertEqual(forward.upper(), backward.upper())


class TestLeafColor(unittest.TestCase):
    """Tests for the per-leaf palette interpolation."""

    __slots__ = ()

    def test_first_leaf_matches_low_anchor(self):
        """Leaf 0 of n>=2 leaves takes the low anchor hex."""
        palette = ("#D91616", "#D9B816", "#16D916")
        color = sigma._palette._leaf_color(0, 4, palette)
        self.assertEqual(color.upper(), "#D91616")

    def test_last_leaf_matches_high_anchor(self):
        """The last leaf of n>=2 leaves takes the high anchor hex."""
        palette = ("#D91616", "#D9B816", "#16D916")
        color = sigma._palette._leaf_color(3, 4, palette)
        self.assertEqual(color.upper(), "#16D916")

    def test_single_leaf_takes_low_anchor(self):
        """For n_leaves==1 the only leaf gets the low anchor."""
        palette = ("#D91616", "#D9B816", "#16D916")
        color = sigma._palette._leaf_color(0, 1, palette)
        self.assertEqual(color.upper(), "#D91616")

    def test_custom_palette_overrides_default(self):
        """A custom palette tuple drives the endpoints of the per-leaf scale."""
        palette = ("#000080", "#808080", "#FF8000")
        low = sigma._palette._leaf_color(0, 3, palette)
        high = sigma._palette._leaf_color(2, 3, palette)
        self.assertEqual(low.upper(), "#000080")
        self.assertEqual(high.upper(), "#FF8000")


class TestContrastForeground(unittest.TestCase):
    """Tests for the WCAG-2 contrast foreground selector."""

    __slots__ = ()

    def test_dark_background_picks_white_text(self):
        """A near-black hex background pairs with white foreground text."""
        result = sigma._palette._contrast_foreground("#000000")
        self.assertEqual(result, "white")

    def test_light_background_picks_black_text(self):
        """A near-white hex background pairs with black foreground text."""
        result = sigma._palette._contrast_foreground("#FFFFFF")
        self.assertEqual(result, "black")


if __name__ == "__main__":
    unittest.main()
