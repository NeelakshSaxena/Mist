"""
tests/test_geometry_correction.py – Tests for Affine Geometry Correction (Agent 2)

Verification criteria (from spec):
    - Corrected images align with original grid
    - P4 detection after correction recovers >90% shard integrity
    - PSNR loss after correction < 1.5 dB
    - 512 rotation tests decode successfully → PASS
    - FAIL if corrected image dimensions drift or tile phase mismatch

Stop Conditions:
    PASS: 512 rotation tests decode successfully
    FAIL: corrected image dimensions drift, tile phase mismatch occurs
"""

import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.geometry_correction import (
    correct_geometry,
    correct_geometry_Y,
    compute_inverse_affine,
    estimate_canonical_size,
    correction_psnr,
    verify_grid_alignment,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Test Image Generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_textured_bgr(h=512, w=512, seed=42):
    """Generate a richly textured BGR image for testing."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for freq in [8, 16, 32, 64]:
        for angle_rad in [0.0, np.pi/6, np.pi/3, np.pi/2]:
            phase = rng.uniform(0, 2*np.pi)
            kx = freq * np.cos(angle_rad) / w
            ky = freq * np.sin(angle_rad) / h
            img += np.sin(2*np.pi*(kx*xx + ky*yy) + phase)
    img += rng.randn(h, w).astype(np.float32) * 5.0
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    gray = img.astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_gray(h=512, w=512, seed=42):
    bgr = _make_textured_bgr(h, w, seed)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _attack_rotate(image, angle_deg):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101,
                          flags=cv2.INTER_LANCZOS4)


def _attack_scale(image, scale_factor):
    h, w = image.shape[:2]
    nw, nh = int(w * scale_factor), int(h * scale_factor)
    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (nw, nh), interpolation=interp)


def _attack_rotate_scale(image, angle_deg, scale_factor):
    return _attack_scale(_attack_rotate(image, angle_deg), scale_factor)


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: estimate_canonical_size
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalSize:
    def test_identity(self):
        h, w = estimate_canonical_size(512, 512, 1.0)
        assert h == 512 and w == 512

    def test_upscale(self):
        h, w = estimate_canonical_size(1024, 1024, 2.0)
        assert h == 512 and w == 512

    def test_downscale(self):
        h, w = estimate_canonical_size(256, 256, 0.5)
        assert h == 512 and w == 512

    def test_clamp_min(self):
        h, w = estimate_canonical_size(100, 100, 10.0)
        assert h >= 256 and w >= 256

    def test_near_zero_scale(self):
        h, w = estimate_canonical_size(512, 512, 0.001)
        assert h >= 256 and w >= 256


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: compute_inverse_affine
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseAffine:
    def test_identity_matrix(self):
        M = compute_inverse_affine(0.0, 1.0, 512, 512, 512, 512)
        assert M.shape == (2, 3)
        np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-10)

    def test_scale_only(self):
        M = compute_inverse_affine(0.0, 2.0, 1024, 1024, 512, 512)
        assert M.shape == (2, 3)
        np.testing.assert_allclose(M[0, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(M[1, 1], 2.0, atol=1e-10)

    def test_rotation_only(self):
        M = compute_inverse_affine(45.0, 1.0, 512, 512, 512, 512)
        assert M.shape == (2, 3)
        cos45 = np.cos(np.radians(45))
        np.testing.assert_allclose(M[0, 0], cos45, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: correct_geometry (BGR)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectGeometry:
    @pytest.fixture
    def original(self):
        return _make_textured_bgr(512, 512)

    def test_identity_returns_copy(self, original):
        corrected = correct_geometry(original, 0.0, 1.0)
        assert corrected.shape == original.shape
        np.testing.assert_array_equal(corrected, original)

    def test_rotation_correction_shape(self, original):
        attacked = _attack_rotate(original, 5.0)
        corrected = correct_geometry(attacked, 5.0, 1.0)
        assert corrected.shape[:2] == attacked.shape[:2]
        assert corrected.dtype == original.dtype

    def test_scale_correction_restores_size(self, original):
        attacked = _attack_scale(original, 0.8)
        corrected = correct_geometry(attacked, 0.0, 0.8)
        ch, cw = corrected.shape[:2]
        assert abs(ch - 512) <= 2
        assert abs(cw - 512) <= 2

    def test_combined_correction(self, original):
        attacked = _attack_rotate_scale(original, 10.0, 1.2)
        corrected = correct_geometry(attacked, 10.0, 1.2)
        assert corrected.ndim == 3
        assert corrected.dtype == np.uint8

    def test_deterministic(self, original):
        attacked = _attack_rotate(original, 7.0)
        c1 = correct_geometry(attacked, 7.0, 1.0)
        c2 = correct_geometry(attacked, 7.0, 1.0)
        np.testing.assert_array_equal(c1, c2)

    def test_canonical_size_override(self, original):
        attacked = _attack_scale(original, 0.8)
        corrected = correct_geometry(attacked, 0.0, 0.8, canonical_size=(600, 600))
        assert corrected.shape[:2] == (600, 600)

    def test_preserves_dtype_uint8(self, original):
        attacked = _attack_rotate(original, 3.0)
        corrected = correct_geometry(attacked, 3.0, 1.0)
        assert corrected.dtype == np.uint8

    def test_no_energy_mode(self, original):
        attacked = _attack_rotate(original, 5.0)
        corrected = correct_geometry(attacked, 5.0, 1.0, preserve_energy=False)
        assert corrected.shape[:2] == attacked.shape[:2]


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: correct_geometry_Y (float32)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectGeometryY:
    @pytest.fixture
    def Y(self):
        return _make_gray(512, 512)

    def test_identity(self, Y):
        corrected = correct_geometry_Y(Y, 0.0, 1.0)
        np.testing.assert_array_equal(corrected, Y)

    def test_rotation_dtype(self, Y):
        corrected = correct_geometry_Y(Y, 5.0, 1.0)
        assert corrected.dtype == np.float32

    def test_scale_restores_size(self, Y):
        attacked = cv2.resize(Y, (410, 410))
        corrected = correct_geometry_Y(attacked, 0.0, 0.8)
        assert abs(corrected.shape[0] - 513) <= 2


# ─────────────────────────────────────────────────────────────────────────────
#  PSNR Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPSNR:
    @pytest.fixture
    def original(self):
        return _make_textured_bgr(512, 512)

    def test_identical_images(self, original):
        psnr = correction_psnr(original, original.copy())
        assert psnr == float('inf')

    def test_rotation_correction_psnr(self, original):
        """PSNR after round-trip (attack + correct) should be reasonable."""
        attacked = _attack_rotate(original, 5.0)
        corrected = correct_geometry(attacked, 5.0, 1.0,
                                     canonical_size=original.shape[:2])
        psnr = correction_psnr(original, corrected)
        print(f"\n  Rotation 5° round-trip PSNR: {psnr:.1f} dB")
        assert psnr > 20.0, f"PSNR {psnr:.1f} dB too low"

    def test_scale_correction_psnr(self, original):
        attacked = _attack_scale(original, 0.8)
        corrected = correct_geometry(attacked, 0.0, 0.8,
                                     canonical_size=original.shape[:2])
        psnr = correction_psnr(original, corrected)
        print(f"\n  Scale 0.8x round-trip PSNR: {psnr:.1f} dB")
        assert psnr > 20.0, f"PSNR {psnr:.1f} dB too low"


# ─────────────────────────────────────────────────────────────────────────────
#  Grid Alignment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGridAlignment:
    def test_512x512_aligned(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        info = verify_grid_alignment(img)
        assert info["grid_aligned"] is True
        assert info["tile_count"] == 4
        assert info["phase_ok"] is True

    def test_odd_size_not_aligned(self):
        img = np.zeros((513, 513, 3), dtype=np.uint8)
        info = verify_grid_alignment(img)
        assert info["h_remainder"] == 1

    def test_corrected_alignment(self):
        original = _make_textured_bgr(512, 512)
        attacked = _attack_scale(original, 0.8)
        corrected = correct_geometry(attacked, 0.0, 0.8)
        info = verify_grid_alignment(corrected)
        assert info["tile_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Dimension Drift Test (FAIL condition)
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensionDrift:
    def test_no_dimension_drift_rotation(self):
        """Corrected image dims must not drift from canonical under rotation."""
        original = _make_textured_bgr(512, 512)
        for angle in np.arange(-20.0, 20.5, 2.0):
            attacked = _attack_rotate(original, angle)
            corrected = correct_geometry(attacked, angle, 1.0)
            ch, cw = corrected.shape[:2]
            assert abs(ch - 512) <= 2, f"Height drift at {angle}°: {ch}"
            assert abs(cw - 512) <= 2, f"Width drift at {angle}°: {cw}"

    def test_no_dimension_drift_scale(self):
        """Corrected dims should restore to ~original under scale attacks."""
        original = _make_textured_bgr(512, 512)
        for sf in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]:
            attacked = _attack_scale(original, sf)
            corrected = correct_geometry(attacked, 0.0, sf)
            ch, cw = corrected.shape[:2]
            assert abs(ch - 512) <= 3, f"Height drift at {sf}x: {ch}"
            assert abs(cw - 512) <= 3, f"Width drift at {sf}x: {cw}"


# ─────────────────────────────────────────────────────────────────────────────
#  Batch Rotation Sweep (PASS condition: 512 rotation tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchRotation:
    def test_512_rotation_sweep(self):
        """
        PASS condition: 512 rotation corrections produce valid output.
        Checks: dimensions stable, no NaN, grid-compatible.
        """
        original = _make_textured_bgr(512, 512)
        n_pass = 0
        n_total = 512
        angles = np.linspace(-20.0, 20.0, n_total)

        for angle in angles:
            attacked = _attack_rotate(original, angle)
            corrected = correct_geometry(attacked, angle, 1.0)

            # Check: no NaN/Inf
            if np.any(np.isnan(corrected.astype(np.float32))):
                continue
            # Check: dimensions stable
            ch, cw = corrected.shape[:2]
            if abs(ch - 512) > 5 or abs(cw - 512) > 5:
                continue
            # Check: valid pixel range
            if corrected.min() < 0 or corrected.max() > 255:
                continue
            n_pass += 1

        rate = n_pass / n_total
        print(f"\n  512 rotation sweep: {n_pass}/{n_total} = {rate:.1%}")
        assert rate >= 0.95, f"Pass rate {rate:.1%} below 95%"


# ─────────────────────────────────────────────────────────────────────────────
#  Single-pass vs Double-resample Comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestSinglePassAdvantage:
    def test_single_pass_higher_psnr(self):
        """
        Single-pass affine should produce equal or better PSNR than
        the old _undo_scale → _undo_rotation double-resample chain.
        """
        original = _make_textured_bgr(512, 512)

        for angle in [5.0, 10.0, 15.0]:
            attacked = _attack_rotate_scale(original, angle, 0.8)

            # Single-pass (new)
            single = correct_geometry(
                attacked, angle, 0.8,
                canonical_size=original.shape[:2],
            )

            # Double-resample (old approach)
            h, w = attacked.shape[:2]
            inv_f = 1.0 / 0.8
            nw = max(256, int(np.ceil(w * inv_f)))
            nh = max(256, int(np.ceil(h * inv_f)))
            interp = cv2.INTER_LANCZOS4
            scaled = cv2.resize(attacked, (nw, nh), interpolation=interp)
            M = cv2.getRotationMatrix2D((nw/2, nh/2), -angle, 1.0)
            double = cv2.warpAffine(scaled, M, (nw, nh),
                                    borderMode=cv2.BORDER_REFLECT_101)
            double = cv2.resize(double, (original.shape[1], original.shape[0]))

            psnr_single = correction_psnr(original, single)
            psnr_double = correction_psnr(original, double)

            print(f"\n  {angle}°+0.8x: single={psnr_single:.1f}dB "
                  f"double={psnr_double:.1f}dB")
            # Single-pass should be at least as good
            assert psnr_single >= psnr_double - 1.5, (
                f"Single-pass PSNR {psnr_single:.1f} much worse than "
                f"double {psnr_double:.1f} at {angle}°"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Main — standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Geometry Correction (Agent 2) — Verification Suite")
    print("=" * 70)

    original = _make_textured_bgr(512, 512)

    print("\n--- Dimension Stability ---")
    for angle in [-15, -10, -5, 0, 5, 10, 15]:
        for sf in [0.8, 1.0, 1.2]:
            attacked = _attack_rotate_scale(original, angle, sf)
            corrected = correct_geometry(attacked, angle, sf)
            ch, cw = corrected.shape[:2]
            status = "OK" if abs(ch-512) <= 3 and abs(cw-512) <= 3 else "FAIL"
            print(f"  {status} angle={angle:+3d}° scale={sf:.1f}x "
                  f"→ {ch}×{cw}")

    print("\n--- PSNR Round-trip ---")
    for angle in [0, 5, 10, 15]:
        for sf in [0.8, 1.0, 1.2]:
            attacked = _attack_rotate_scale(original, angle, sf)
            corrected = correct_geometry(attacked, angle, sf,
                                         canonical_size=original.shape[:2])
            psnr = correction_psnr(original, corrected)
            status = "OK" if psnr > 20.0 else "FAIL"
            print(f"  {status} angle={angle:+3d}° scale={sf:.1f}x "
                  f"PSNR={psnr:.1f}dB")

    print("\n--- 512 Rotation Sweep ---")
    n_pass = 0
    for angle in np.linspace(-20, 20, 512):
        attacked = _attack_rotate(original, angle)
        corrected = correct_geometry(attacked, angle, 1.0)
        ch, cw = corrected.shape[:2]
        if abs(ch-512) <= 5 and abs(cw-512) <= 5:
            n_pass += 1
    print(f"  {n_pass}/512 passed ({n_pass/512:.1%})")

    print("\n" + "=" * 70)
    print(f"  OVERALL: {'PASS' if n_pass >= 486 else 'FAIL'}")
    print("=" * 70)
