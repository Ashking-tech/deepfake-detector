"""Step-1 tests: prove the decision calculator mixes opinions correctly.

Canonical vector (frozen in README Step 0 — change the README first if you
ever touch these numbers): stripped history (all unsure) + erased watermark
(weak real-lean) + strong picture-detective (fake-lean) must fuse to
auth=0.109375, synth=0.791667, uncert=0.098958, K=0.04 -> Suspicious.
"""

import time

import pytest

from app.fusion.classify import (
    PROVENANCE_AVAILABLE,
    SUSPICIOUS,
    UNKNOWN,
    VERIFIED_AI_ORIGIN,
    classify,
)
from app.fusion.dempster import (
    belief_synth,
    combine,
    plausibility_synth,
    sequential_fuse,
)
from app.fusion.mass import MassFunction, discount, vacuous


# The frozen evasion example. Guards the exact product table:
# 3 products land on REAL (0.005 + 0.005 + 0.095),
# 1 lands on FAKE (0.760),
# 1 lands on UNSURE (0.095),
# 1 lands on CONFLICT (0.040).
# docs.md once dropped two REAL terms and got auth=0.005 —
# this test would have caught that.
def test_canonical_evasion_vector():
    """Stripped manifest + erased watermark + strong forensics."""
    # Step 1: build the three input opinions, one line each.
    m1 = MassFunction(m_auth=0.0, m_synth=0.0, m_uncert=1.0)
    m2 = MassFunction(m_auth=0.05, m_synth=0.0, m_uncert=0.95)
    m3 = MassFunction(m_auth=0.10, m_synth=0.80, m_uncert=0.10)

    # Step 2: fuse them all.
    inputs = [m1, m2, m3]
    fused, k = sequential_fuse(inputs)

    # Step 3: check the conflict level.
    assert k == pytest.approx(0.04, abs=1e-9)

    # Step 4: check each fused share on its own line.
    assert fused.m_auth == pytest.approx(0.109375, abs=1e-6)
    assert fused.m_synth == pytest.approx(0.791667, abs=1e-6)
    assert fused.m_uncert == pytest.approx(0.098958, abs=1e-6)

    # Step 5: check the low/high fake bounds.
    low = belief_synth(fused)
    assert low == pytest.approx(0.791667, abs=1e-6)
    high = plausibility_synth(fused)
    assert high == pytest.approx(0.890625, abs=1e-6)

    # Step 6: check the final verdict.
    verdict = classify(fused, driver="forensics", k=k)
    assert verdict == SUSPICIOUS


# Mixing with "I don't know" must change nothing (neutral element).
# If this breaks, the stripped-manifest path would distort every verdict.
def test_vacuous_is_neutral():
    """m1 all-unsure fused with m2 must equal m2."""
    # Step 1: build the inputs.
    m1 = vacuous()
    m2 = MassFunction(m_auth=0.05, m_synth=0.0, m_uncert=0.95)

    # Step 2: mix them.
    fused, k = combine(m1, m2)

    # Step 3: check nothing changed, one assert per share.
    assert k == pytest.approx(0.0, abs=1e-9)
    assert fused.m_auth == pytest.approx(0.05, abs=1e-9)
    assert fused.m_synth == pytest.approx(0.0, abs=1e-9)
    assert fused.m_uncert == pytest.approx(0.95, abs=1e-9)


# Flat contradiction (crypto says REAL 0.95, hidden code says FAKE 0.90):
# K = 0.95*0.90 = 0.855 > 0.6, so Yager parks conflict in unsure and
# the verdict must be Unknown — never a confident coin flip (Zadeh paradox).
def test_total_conflict_abstains():
    """Directly contradictory proofs must yield Unknown, not certainty."""
    # Step 1: build two opinions that disagree completely.
    m1 = MassFunction(m_auth=0.95, m_synth=0.0, m_uncert=0.05)
    m2 = MassFunction(m_auth=0.0, m_synth=0.90, m_uncert=0.10)

    # Step 2: mix them.
    fused, k = combine(m1, m2)

    # Step 3: conflict must be huge ...
    assert k == pytest.approx(0.855, abs=1e-9)

    # Step 4: ... so almost everything sits in unsure ...
    assert fused.m_uncert == pytest.approx(0.86, abs=1e-9)

    # Step 5: ... and the verdict abstains.
    verdict = classify(fused, driver="provenance", k=k)
    assert verdict == UNKNOWN


# A lone cryptographic proof must pass through untouched and map to the
# proof statuses (no dilution by nothing).
def test_single_crypto_proof_statuses():
    """Valid AI manifest -> Verified AI Origin; valid camera -> Provenance."""
    # Step 1: an AI-proof opinion with a crypto driver is proof.
    ai = MassFunction(m_auth=0.0, m_synth=0.95, m_uncert=0.05)
    verdict_ai = classify(ai, driver="provenance")
    assert verdict_ai == VERIFIED_AI_ORIGIN

    # Step 2: a camera-proof opinion with a crypto driver is proof.
    cam = MassFunction(m_auth=0.95, m_synth=0.0, m_uncert=0.05)
    verdict_cam = classify(cam, driver="provenance")
    assert verdict_cam == PROVENANCE_AVAILABLE

    # Step 3: same numbers WITHOUT a crypto driver are only a hint.
    verdict_hint = classify(ai, driver="forensics")
    assert verdict_hint == SUSPICIOUS


# Garbage opinions must be rejected at creation, and distrust must move
# points into unsure (discounting used to tame overconfident checkers).
def test_validation_and_discount():
    """Bad masses raise; discount(alpha) shifts weight to unsure."""
    # Step 1: shares adding to 1.5 must be rejected.
    with pytest.raises(ValueError):
        MassFunction(m_auth=0.5, m_synth=0.5, m_uncert=0.5)

    # Step 2: build a confident opinion.
    m = MassFunction(m_auth=0.10, m_synth=0.80, m_uncert=0.10)

    # Step 3: distrust it 50% ...
    d = discount(m, 0.5)

    # Step 4: ... so confident shares halve and unsure grows.
    assert d.m_synth == pytest.approx(0.40, abs=1e-9)
    assert d.m_auth == pytest.approx(0.05, abs=1e-9)
    assert d.m_uncert == pytest.approx(0.55, abs=1e-9)


# Fusion must be instant (<10ms for 3 opinions) or the 10s/image budget in
# NFR-4 would be eaten by math instead of models.
def test_fusion_is_fast():
    """3-opinion fuse completes in under 10ms."""
    # Step 1: build the inputs.
    m1 = MassFunction(m_auth=0.0, m_synth=0.0, m_uncert=1.0)
    m2 = MassFunction(m_auth=0.05, m_synth=0.0, m_uncert=0.95)
    m3 = MassFunction(m_auth=0.10, m_synth=0.80, m_uncert=0.10)
    inputs = [m1, m2, m3]

    # Step 2: time the fuse.
    start = time.perf_counter()
    sequential_fuse(inputs)
    elapsed = time.perf_counter() - start

    # Step 3: check it was fast.
    assert elapsed < 0.01
