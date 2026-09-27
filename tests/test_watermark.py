"""Step-3 tests: prove the hidden-code checker finds our stamp honestly.

Two groups: pure mapping tests (run ALWAYS, no library needed) and
detector tests (skipped until invisible-watermark + opencv are installed,
per the "code first, install later" decision — the suite must stay green
before the heavy downloads land).
"""

import importlib.util

import pytest

from app.fusion.classify import SUSPICIOUS, UNKNOWN, VERIFIED_AI_ORIGIN, classify
from app.pillars.watermark.base import get_detector
from app.pillars.watermark.detect import (
    PAYLOAD_ASCII,
    PAYLOAD_BITS,
    detect,
    payload_bits,
    to_mass,
)
from app.pillars.watermark.rivagan_detector import RivaGanDetector
from app.pillars.watermark.base import WatermarkResult
from tests.conftest import fixture_path


# True when the watermark stack is importable (lib + opencv present).
def _lib_available():
    """Check invisible-watermark and cv2 import without importing them."""
    has_lib = importlib.util.find_spec("imwatermark") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None
    return has_lib and has_cv2


# Skip mark for every test that touches real pixels + the real library.
needs_lib = pytest.mark.skipif(
    not _lib_available(),
    reason="needs invisible-watermark + opencv installed",
)


# Our magic must be 8 ASCII chars = exactly 64 bits (dwtDct's payload).
def test_payload_is_64_bits():
    """payload_bits() returns 64 binary ints for 'DEPAUTH1'."""
    # Step 1: build the bits.
    bits = payload_bits()

    # Step 2: check the length.
    assert len(bits) == PAYLOAD_BITS
    assert len(bits) == 64

    # Step 3: check every entry is a clean 0 or 1.
    for bit in bits:
        assert bit == 0 or bit == 1

    # Step 4: check the magic text itself survived refactors.
    assert PAYLOAD_ASCII == "DEPAUTH1"


# A perfectly decoded stamp maps to 90% AI (README's "verified" mass).
def test_to_mass_found_exact():
    """BER 0 found -> (0, 0.90, 0.10)."""
    # Step 1: fabricate a perfect finding (no files needed).
    result = WatermarkResult(found=True, ber=0.0, reason="test", detector="dwtDct")

    # Step 2: map it.
    mass = to_mass(result)

    # Step 3: check each share on its own line.
    assert mass.m_auth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.90, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.10, abs=1e-9)

    # Step 4: as the driving signal this is hard proof.
    verdict = classify(mass, driver="watermark")
    assert verdict == VERIFIED_AI_ORIGIN


# Damage must demote confidence through discount(): BER 0.2 still verifies,
# BER 0.3 slides from proof down to a hint (Suspicious).
def test_to_mass_found_degraded():
    """BER softens the stamp: 0.2 verifies, 0.3 demotes to Suspicious."""
    # Step 1: light damage (BER 0.2) keeps proof status.
    light = WatermarkResult(found=True, ber=0.2, reason="test", detector="dwtDct")
    light_mass = to_mass(light)
    assert light_mass.m_synth == pytest.approx(0.72, abs=1e-9)
    assert light_mass.m_uncert == pytest.approx(0.28, abs=1e-9)
    assert classify(light_mass, driver="watermark") == VERIFIED_AI_ORIGIN

    # Step 2: heavy damage (BER 0.3) drops below the proof line.
    heavy = WatermarkResult(found=True, ber=0.3, reason="test", detector="dwtDct")
    heavy_mass = to_mass(heavy)
    assert heavy_mass.m_synth == pytest.approx(0.63, abs=1e-9)
    assert classify(heavy_mass, driver="watermark") == SUSPICIOUS


# Absence of OUR code leans weakly real: "not stamped by us" is not
# evidence of authenticity.
def test_to_mass_absent():
    """Absent -> (0.05, 0, 0.95), verdict Unknown even as driver."""
    # Step 1: fabricate an absent finding.
    result = WatermarkResult(found=False, ber=0.5, reason="test", detector="dwtDct")

    # Step 2: map it.
    mass = to_mass(result)

    # Step 3: check each share.
    assert mass.m_auth == pytest.approx(0.05, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.95, abs=1e-9)

    # Step 4: even as driver, this abstains.
    verdict = classify(mass, driver="watermark")
    assert verdict == UNKNOWN


# A BER outside [0,1] is a programmer bug -> loud error, not a silent mass.
def test_to_mass_rejects_bad_ber():
    """BER 1.5 raises ValueError."""
    bad = WatermarkResult(found=True, ber=1.5, reason="test", detector="dwtDct")
    with pytest.raises(ValueError):
        to_mass(bad)


# Unknown detector names must fail early with a helpful message.
def test_registry_rejects_unknown():
    """get_detector('definitely-not-a-detector') raises KeyError."""
    with pytest.raises(KeyError):
        get_detector("definitely-not-a-detector")


# The registered dwtDct factory must build a correctly shaped detector.
@needs_lib
def test_registry_builds_dwt():
    """get_detector('dwtDct') builds with name + 64-bit payload."""
    detector = get_detector("dwtDct")
    assert detector.name == "dwtDct"
    assert detector.payload_bits == 64


# The RivaGAN stub must fail loudly with instructions, never pretend.
def test_rivagan_stub_refuses():
    """RivaGanDetector.detect raises NotImplementedError."""
    stub = RivaGanDetector()
    assert stub.payload_bits == 32
    with pytest.raises(NotImplementedError):
        stub.detect(fixture_path("real_01.jpg"))


# A missing file must not crash: absent finding, fully safe.
def test_missing_file_is_absent():
    """Bogus path -> found False, never an exception."""
    result = detect(fixture_path("does-not-exist.jpg"))
    assert result.found is False


# The stamped JPEG fixture must decode to (near-)zero BER and verify.
@needs_lib
def test_exact_fixture_found():
    """wm_ai_01.jpg -> found, ~90% AI, Verified AI Origin."""
    # Step 1: skip honestly if fixtures aren't generated yet.
    path = fixture_path("wm_ai_01.jpg")
    if not path.exists():
        pytest.skip("watermark fixtures not generated yet")

    # Step 2: run the checker.
    result = detect(path)

    # Step 3: the stamp must be found nearly perfectly.
    assert result.found is True
    assert result.ber <= 0.05

    # Step 4: the opinion must verify.
    mass = to_mass(result)
    assert mass.m_synth >= 0.85
    verdict = classify(mass, driver="watermark")
    assert verdict == VERIFIED_AI_ORIGIN


# The q=60 recompressed copy must still be found (dwtDct passes JPEG),
# with mass no stronger than the exact fixture's (damage never helps).
@needs_lib
def test_jpg60_fixture_found_degraded():
    """wm_ai_01_jpg60.jpg -> found, mass <= exact mass."""
    # Step 1: skip honestly if fixtures aren't generated yet.
    damaged_path = fixture_path("wm_ai_01_jpg60.jpg")
    exact_path = fixture_path("wm_ai_01.jpg")
    if not damaged_path.exists() or not exact_path.exists():
        pytest.skip("watermark fixtures not generated yet")

    # Step 2: run the checker on both.
    damaged_result = detect(damaged_path)
    exact_result = detect(exact_path)

    # Step 3: the damaged copy must still be found...
    assert damaged_result.found is True

    # Step 4: ...but never score HIGHER than the clean copy.
    damaged_mass = to_mass(damaged_result)
    exact_mass = to_mass(exact_result)
    assert damaged_mass.m_synth <= exact_mass.m_synth + 1e-9


# Plain and stripped photos carry no stamp: absent, mostly unsure.
# The stripped case doubles as proof the check doesn't need metadata.
@needs_lib
def test_unstamped_photos_absent():
    """real_01.jpg + stripped_01.jpg -> absent, 95% unsure."""
    for name in ["real_01.jpg", "stripped_01.jpg"]:
        # Step 1: run the checker.
        result = detect(fixture_path(name))

        # Step 2: nothing stamped here.
        assert result.found is False

        # Step 3: opinion leans weakly real, mostly unsure.
        mass = to_mass(result)
        assert mass.m_uncert == pytest.approx(0.95, abs=1e-9)
        assert classify(mass, driver="watermark") == UNKNOWN
