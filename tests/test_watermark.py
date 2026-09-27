"""Step-3 tests: prove the hidden-code checker finds our stamp honestly.

Two groups: pure mapping tests (run ALWAYS, no library needed) and
detector tests (skipped until invisible-watermark + onnxruntime + opencv
are installed — the suite stays green before the heavy downloads land).

Default detector is RivaGAN (32-bit magic "DEPA"), measured perfect in our
stack. dwtDct stays registered but is documented-broken (constant output),
so no file test uses it.
"""

import importlib.util

import pytest

from app.fusion.classify import SUSPICIOUS, UNKNOWN, VERIFIED_AI_ORIGIN, classify
from app.pillars.watermark.base import WatermarkResult, get_detector
from app.pillars.watermark.detect import (
    PAYLOAD_ASCII,
    PAYLOAD_BITS,
    RIVAGAN_ASCII,
    RIVAGAN_BITS,
    detect,
    payload_bits,
    rivagan_bits,
    to_mass,
)
from app.pillars.watermark.rivagan_detector import MAX_BER_FOR_FOUND
from tests.conftest import fixture_path


# True when the watermark stack is importable (lib + onnx + opencv).
def _lib_available():
    """Check invisible-watermark, onnxruntime and cv2 without importing."""
    has_lib = importlib.util.find_spec("imwatermark") is not None
    has_onnx = importlib.util.find_spec("onnxruntime") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None
    if has_lib and has_onnx and has_cv2:
        return True
    return False


# Skip mark for every test that touches real pixels + the real library.
needs_lib = pytest.mark.skipif(
    not _lib_available(),
    reason="needs invisible-watermark + onnxruntime + opencv installed",
)


# Skip mark for tests needing generated fixtures on disk.
def _fixtures_ready():
    """True when the stamped fixtures exist."""
    wanted = ["wm_ai_01.jpg", "wm_ai_01_jpg60.jpg", "wm_ai_02.png", "wm_ai_01_rot30.jpg"]
    for name in wanted:
        if not fixture_path(name).exists():
            return False
    return True


needs_fixtures = pytest.mark.skipif(
    not _fixtures_ready(),
    reason="watermark fixtures not generated yet",
)


# Our live magic must be 4 ASCII chars = exactly 32 bits (RivaGAN's size).
def test_rivagan_magic_is_32_bits():
    """rivagan_bits() returns 32 binary ints for 'DEPA'."""
    # Step 1: build the bits.
    bits = rivagan_bits()

    # Step 2: check the length.
    assert len(bits) == RIVAGAN_BITS
    assert len(bits) == 32

    # Step 3: check every entry is a clean 0 or 1.
    for bit in bits:
        assert bit == 0 or bit == 1

    # Step 4: check the magic text itself survived refactors.
    assert RIVAGAN_ASCII == "DEPA"


# The legacy 64-bit magic stays intact for the record (dwtDct's payload).
def test_legacy_magic_is_64_bits():
    """payload_bits() still returns 64 binary ints for 'DEPAUTH1'."""
    bits = payload_bits()
    assert len(bits) == PAYLOAD_BITS
    assert len(bits) == 64
    assert PAYLOAD_ASCII == "DEPAUTH1"


# The default detector must be RivaGAN (measured perfect, fixes crop).
def test_default_detector_is_rivagan():
    """detect() without a name uses rivaGan."""
    import inspect

    import app.pillars.watermark.detect as detect_module

    assert detect_module.DEFAULT_DETECTOR == "rivaGan"
    sig = inspect.signature(detect_module.detect)
    assert sig.parameters["detector_name"].default == "rivaGan"


# A perfectly decoded stamp maps to 90% AI (README's "verified" mass).
def test_to_mass_found_exact():
    """BER 0 found -> (0, 0.90, 0.10)."""
    # Step 1: fabricate a perfect finding (no files needed).
    result = WatermarkResult(found=True, ber=0.0, reason="test", detector="rivaGan")

    # Step 2: map it.
    mass = to_mass(result)

    # Step 3: check each share on its own line.
    assert mass.m_auth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.90, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.10, abs=1e-9)

    # Step 4: as the driving signal this is hard proof.
    verdict = classify(mass, driver="watermark")
    assert verdict == VERIFIED_AI_ORIGIN


# Damage must demote confidence through discount(): BER 0.1 still verifies,
# BER 0.3 slides from proof down to a hint (Suspicious).
def test_to_mass_found_degraded():
    """BER softens the stamp: 0.1 verifies, 0.3 demotes to Suspicious."""
    # Step 1: light damage (BER 0.1) keeps proof status.
    light = WatermarkResult(found=True, ber=0.1, reason="test", detector="rivaGan")
    light_mass = to_mass(light)
    assert light_mass.m_synth == pytest.approx(0.81, abs=1e-9)
    assert light_mass.m_uncert == pytest.approx(0.19, abs=1e-9)
    assert classify(light_mass, driver="watermark") == VERIFIED_AI_ORIGIN

    # Step 2: heavy damage (BER 0.3) drops below the proof line.
    heavy = WatermarkResult(found=True, ber=0.3, reason="test", detector="rivaGan")
    heavy_mass = to_mass(heavy)
    assert heavy_mass.m_synth == pytest.approx(0.63, abs=1e-9)
    assert classify(heavy_mass, driver="watermark") == SUSPICIOUS


# Absence of OUR code leans weakly real: "not stamped by us" is not
# evidence of authenticity.
def test_to_mass_absent():
    """Absent -> (0.05, 0, 0.95), verdict Unknown even as driver."""
    # Step 1: fabricate an absent finding.
    result = WatermarkResult(found=False, ber=0.4, reason="test", detector="rivaGan")

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
    bad = WatermarkResult(found=True, ber=1.5, reason="test", detector="rivaGan")
    with pytest.raises(ValueError):
        to_mass(bad)


# Unknown detector names must fail early with a helpful message.
def test_registry_rejects_unknown():
    """get_detector('definitely-not-a-detector') raises KeyError."""
    with pytest.raises(KeyError):
        get_detector("definitely-not-a-detector")


# Both registered factories must build correctly shaped detectors.
@needs_lib
def test_registry_builds_detectors():
    """rivaGan builds 32-bit, dwtDct builds 64-bit (legacy record)."""
    # Step 1: the live default builds.
    live = get_detector("rivaGan")
    assert live.name == "rivaGan"
    assert live.payload_bits == 32

    # Step 2: the retired reader still constructs (documented-broken,
    # never default, kept so the failure stays on record).
    legacy = get_detector("dwtDct")
    assert legacy.name == "dwtDct"
    assert legacy.payload_bits == 64


# The RivaGAN threshold must match the module constant (single source).
def test_threshold_constant():
    """Found gate is MAX_BER_FOR_FOUND = 0.15."""
    assert MAX_BER_FOR_FOUND == pytest.approx(0.15, abs=1e-9)


# A missing file must not crash: absent finding, fully safe.
def test_missing_file_is_absent():
    """Bogus path -> found False, never an exception."""
    result = detect(fixture_path("does-not-exist.jpg"))
    assert result.found is False


# The stamped JPEG fixtures must decode to zero BER and verify.
@needs_lib
@needs_fixtures
def test_exact_fixtures_found():
    """wm_ai_01.jpg + wm_ai_02.png -> found, BER 0, Verified AI Origin."""
    for name in ["wm_ai_01.jpg", "wm_ai_02.png"]:
        # Step 1: run the checker.
        result = detect(fixture_path(name))

        # Step 2: the stamp must be found perfectly.
        assert result.found is True
        assert result.ber == pytest.approx(0.0, abs=1e-9)

        # Step 3: the opinion must verify.
        mass = to_mass(result)
        assert mass.m_synth == pytest.approx(0.90, abs=1e-9)
        verdict = classify(mass, driver="watermark")
        assert verdict == VERIFIED_AI_ORIGIN


# The q=60 recompressed copy must still be found (RivaGAN passes JPEG),
# with mass no stronger than the exact fixture's (damage never helps).
@needs_lib
@needs_fixtures
def test_jpg60_fixture_found():
    """wm_ai_01_jpg60.jpg -> found, mass <= exact mass."""
    # Step 1: run the checker on both.
    damaged_result = detect(fixture_path("wm_ai_01_jpg60.jpg"))
    exact_result = detect(fixture_path("wm_ai_01.jpg"))

    # Step 2: the damaged copy must still be found...
    assert damaged_result.found is True

    # Step 3: ...but never score HIGHER than the clean copy.
    damaged_mass = to_mass(damaged_result)
    exact_mass = to_mass(exact_result)
    assert damaged_mass.m_synth <= exact_mass.m_synth + 1e-9


# The rotated copy must read absent: rotation kills neural watermarks,
# and the checker must say "unsure", never invent a positive.
@needs_lib
@needs_fixtures
def test_rot30_fixture_absent():
    """wm_ai_01_rot30.jpg -> absent, 95% unsure."""
    # Step 1: run the checker.
    result = detect(fixture_path("wm_ai_01_rot30.jpg"))

    # Step 2: the stamp is destroyed -> absent...
    assert result.found is False

    # Step 3: ...with damage visible in the BER...
    assert result.ber > 0.15

    # Step 4: ...and an honest unsure opinion.
    mass = to_mass(result)
    assert mass.m_uncert == pytest.approx(0.95, abs=1e-9)
    assert classify(mass, driver="watermark") == UNKNOWN


# Plain, stripped, and synthetic-placeholder photos carry no stamp: absent,
# mostly unsure. The stripped case doubles as proof the check needs no
# metadata; the synthetic case proves pathological content doesn't fake
# a positive.
@needs_lib
def test_unstamped_photos_absent():
    """real_01 + stripped_01 + synthetic_01 -> absent, 95% unsure."""
    for name in ["real_01.jpg", "stripped_01.jpg", "synthetic_01.png"]:
        # Step 1: run the checker.
        result = detect(fixture_path(name))

        # Step 2: nothing stamped here.
        assert result.found is False

        # Step 3: opinion leans weakly real, mostly unsure.
        mass = to_mass(result)
        assert mass.m_uncert == pytest.approx(0.95, abs=1e-9)
        assert classify(mass, driver="watermark") == UNKNOWN
