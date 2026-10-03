"""Step-4 tests: prove the picture detective measures honestly.

What is proven here: the FFT slope separates OUR fixtures (placeholder
vs photos), scores map to capped masses (hint, never proof), real photos
yield Unknown (lean, not verdict), and the deep backbones run with shapes
but vote NOTHING until Phase 7 trains their heads (zero-weight guards).
"""

import importlib.util

import numpy as np
import pytest

from app.fusion.classify import SUSPICIOUS, UNKNOWN, classify
from app.pillars.forensics import fft_branch, preprocess
from app.pillars.forensics.detect import (
    BASE_UNCERTAINTY,
    CONVNEXT_WEIGHT,
    DINO_WEIGHT,
    ForensicsResult,
    check,
    to_mass,
)
from tests.conftest import fixture_path


# True when the deep stacks are importable (timm + transformers present).
def _backbones_available():
    """Check timm and transformers without importing them."""
    has_timm = importlib.util.find_spec("timm") is not None
    has_tf = importlib.util.find_spec("transformers") is not None
    if has_timm and has_tf:
        return True
    return False


# Skip mark for tests that load the ~460MB deep backbones.
needs_backbones = pytest.mark.skipif(
    not _backbones_available(),
    reason="needs timm + transformers installed",
)


# The shared preparation must output a 224x224 RGB square, always.
def test_prepare_shapes():
    """real_01.jpg -> 224x224 RGB square, array in [0,1]."""
    # Step 1: load the photo.
    image = preprocess.load_image(fixture_path("real_01.jpg"))
    assert image is not None

    # Step 2: prepare it.
    small = preprocess.prepare(image)

    # Step 3: check the shape, one assert per fact.
    assert small.size == (224, 224)
    assert small.mode == "RGB"

    # Step 4: the array form is scaled to [0,1].
    arr = preprocess.to_array(small)
    assert arr.shape == (224, 224, 3)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


# A wide picture must be center-cropped to its height (no squishing).
def test_square_crop_wide_image():
    """1024x768 photo -> 768x768 square before resize."""
    # Step 1: load (real_01 is 1024 wide, 768 tall).
    image = preprocess.load_image(fixture_path("real_01.jpg"))

    # Step 2: crop only (no resize yet).
    square = preprocess.square_crop(image)

    # Step 3: the square side equals the shorter side.
    assert square.size == (768, 768)


# Missing files must yield None, never an exception.
def test_load_missing_returns_none():
    """Bogus path -> None."""
    assert preprocess.load_image(fixture_path("does-not-exist.jpg")) is None


# The slope must separate OUR placeholder (flat spectrum) from photos
# (steep spectra). This is fixture separation, not general AI detection.
def test_slope_separates_fixtures():
    """synthetic slope << real slopes (measured: 1.6 vs 7.3+)."""
    # Step 1: measure all three through the real pipeline path.
    slopes = {}
    for name in ["real_01.jpg", "real_02.png", "synthetic_01.png"]:
        image = preprocess.load_image(fixture_path(name))
        small = preprocess.prepare(image)
        gray = preprocess.to_array(small).mean(axis=2)
        slopes[name] = fft_branch.spectral_slope(gray)

    # Step 2: the placeholder sits far below both photos.
    assert slopes["synthetic_01.png"] < 2.5
    assert slopes["real_01.jpg"] > 6.0
    assert slopes["real_02.png"] > 6.0


# The logistic map must be monotonic and bounded: lower slope -> higher
# P(fake), always inside [0,1].
def test_slope_to_score_monotonic():
    """score(1.5) > score(4.0) > score(8.0), all in [0,1]."""
    # Step 1: score three slopes.
    low = fft_branch.slope_to_score(1.5)
    mid = fft_branch.slope_to_score(4.0)
    high = fft_branch.slope_to_score(8.0)

    # Step 2: check order and bounds, one line each.
    assert 0.0 <= high <= 1.0
    assert 0.0 <= mid <= 1.0
    assert 0.0 <= low <= 1.0
    assert low > mid
    assert mid > high

    # Step 3: the threshold midpoint must be exactly unsure.
    assert mid == pytest.approx(0.5, abs=1e-9)


# The ambiguity zone flags doubtful middles and trusts clear extremes.
def test_ambiguity_zone():
    """4.0 ambiguous; 7.9 and 1.5 not."""
    assert fft_branch.is_ambiguous(4.0) is True
    assert fft_branch.is_ambiguous(7.9) is False
    assert fft_branch.is_ambiguous(1.5) is False


# Masses must split the confident share by score and sum to exactly 1.
def test_to_mass_splits_confidence():
    """score 0.9, unc 0.4 -> (0.06, 0.54, 0.4)."""
    # Step 1: fabricate a strong-fake finding.
    result = ForensicsResult(score=0.9, uncertainty=0.4, reason="test")

    # Step 2: map it.
    mass = to_mass(result)

    # Step 3: check each share on its own line.
    assert mass.m_synth == pytest.approx(0.54, abs=1e-9)
    assert mass.m_auth == pytest.approx(0.06, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.4, abs=1e-9)

    # Step 4: the shares always sum to exactly 1.
    total = mass.m_auth + mass.m_synth + mass.m_uncert
    assert total == pytest.approx(1.0, abs=1e-9)


# Out-of-range inputs are programmer bugs -> loud errors, never silent.
def test_to_mass_rejects_garbage():
    """score 1.5 and uncertainty -0.1 both raise ValueError."""
    # Step 1: bad score fails.
    bad_score = ForensicsResult(score=1.5, uncertainty=0.4, reason="test")
    with pytest.raises(ValueError):
        to_mass(bad_score)

    # Step 2: bad uncertainty fails.
    bad_unc = ForensicsResult(score=0.5, uncertainty=-0.1, reason="test")
    with pytest.raises(ValueError):
        to_mass(bad_unc)


# The placeholder must read strongly fake — but only Suspicious (hint),
# never proof: the 0.40 uncertainty floor caps m_synth at 0.60.
def test_synthetic_reads_suspicious():
    """synthetic_01.png -> score > 0.8, verdict Suspicious, never proof."""
    # Step 1: run the full checker.
    result = check(fixture_path("synthetic_01.png"))

    # Step 2: the heuristic fires strongly.
    assert result.score > 0.8

    # Step 3: the opinion leans fake but stays capped.
    mass = to_mass(result)
    assert mass.m_synth > 0.5
    assert mass.m_synth <= 0.60 + 1e-9

    # Step 4: forensics-driven verdict is a hint, not proof.
    verdict = classify(mass, driver="forensics")
    assert verdict == SUSPICIOUS


# A real photo must lean real yet abstain: the heuristic cannot PROVE
# authenticity either. Unknown here is honesty, not failure.
def test_real_photo_abstains():
    """real_01.jpg -> score < 0.1, verdict Unknown."""
    # Step 1: run the full checker.
    result = check(fixture_path("real_01.jpg"))

    # Step 2: the heuristic leans real.
    assert result.score < 0.1

    # Step 3: but the verdict abstains (no proof either way).
    mass = to_mass(result)
    verdict = classify(mass, driver="forensics")
    assert verdict == UNKNOWN


# JPEG recompression must barely move the score (slope is re-encode
# robust — measured < 0.1 slope shift at q60). Damage must not flip hints.
def test_recompression_is_stable():
    """|score(q60) - score(orig)| < 0.05."""
    # Step 1: run the checker on both.
    clean = check(fixture_path("real_01.jpg"))
    damaged = check(fixture_path("recompressed_01.jpg"))

    # Step 2: the scores must nearly match.
    shift = abs(damaged.score - clean.score)
    assert shift < 0.05


# A missing file must not crash: fully unsure, verdict Unknown.
def test_missing_file_fully_unsure():
    """Bogus path -> uncertainty 1.0, verdict Unknown."""
    # Step 1: run the checker on nothing.
    result = check(fixture_path("does-not-exist.jpg"))

    # Step 2: total ignorance...
    assert result.uncertainty == pytest.approx(1.0, abs=1e-9)

    # Step 3: ...maps to total ignorance and abstains.
    mass = to_mass(result)
    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)
    assert classify(mass, driver="forensics") == UNKNOWN


# GUARD: deep backbones stay silent until Phase 7 trains their heads.
# If anyone enables these weights without dataset training, this test
# forces them to update the test consciously — no silent overclaim.
def test_deep_weights_are_zero():
    """CONVNEXT_WEIGHT and DINO_WEIGHT are 0.0 (wired, not voting)."""
    assert CONVNEXT_WEIGHT == 0.0
    assert DINO_WEIGHT == 0.0
    assert BASE_UNCERTAINTY >= 0.40


# ConvNeXt must emit a finite 768-vector, identical across two calls.
@needs_backbones
def test_convnext_embedding():
    """Real photo -> 768-dim finite vector, deterministic."""
    from app.pillars.forensics import convnext

    # Step 1: prepare pixels once.
    image = preprocess.load_image(fixture_path("real_01.jpg"))
    rgb = preprocess.to_array(preprocess.prepare(image))

    # Step 2: embed twice.
    first = convnext.embed(rgb)
    second = convnext.embed(rgb)

    # Step 3: check shape, finiteness, determinism — one fact per line.
    assert first.shape == (768,)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)


# DINOv2 must emit a finite 768-dim CLS vector, identical across calls;
# the probe stays flagged untrained with its vote inside [0,1].
@needs_backbones
def test_dino_cls_and_probe():
    """Real photo -> 768-dim CLS, deterministic; probe untrained."""
    from app.pillars.forensics import dino_probe

    # Step 1: prepare pixels once.
    image = preprocess.load_image(fixture_path("real_01.jpg"))
    small = preprocess.prepare(image)

    # Step 2: CLS twice.
    first = dino_probe.cls_embed(small)
    second = dino_probe.cls_embed(small)

    # Step 3: shape, finiteness, determinism.
    assert first.shape == (768,)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)

    # Step 4: probe votes but admits it is untrained.
    vote, trained = dino_probe.probe_score(first)
    assert trained is False
    assert 0.0 <= vote <= 1.0
