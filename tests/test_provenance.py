"""Step-2 tests: prove the history-file checker answers honestly.

The golden rule under test: a missing history file must yield total
uncertainty — never "authentic". Platforms strip manifests routinely, so
guessing "real" from absence would be the pipeline's most dangerous lie.
"""

import pytest

from app.fusion.classify import (
    PROVENANCE_AVAILABLE,
    UNKNOWN,
    VERIFIED_AI_ORIGIN,
    classify,
)
from app.pillars.provenance.verify import (
    NO_MANIFEST,
    TAMPERED,
    UNREADABLE,
    VALID_AI,
    VALID_CAMERA,
    ProvenanceResult,
    check,
    to_mass,
)
from tests.conftest import fixture_path


# A normal photo with no history file must come back fully unsure.
def test_plain_photo_has_no_manifest():
    """real_01.jpg -> no_manifest, mass all unsure."""
    # Step 1: run the checker.
    result = check(fixture_path("real_01.jpg"))

    # Step 2: the finding must say "no history file".
    assert result.status == NO_MANIFEST
    assert result.ai_claim is False

    # Step 3: the opinion must be total ignorance.
    mass = to_mass(result)
    assert mass.m_auth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)


# A stripped photo must behave EXACTLY like a plain photo: absence of proof
# is not proof of authenticity. This is the regression guard for the
# pipeline's most dangerous lie.
def test_stripped_photo_never_says_authentic():
    """stripped_01.jpg -> no_manifest, verdict Unknown even as driver."""
    # Step 1: run the checker.
    result = check(fixture_path("stripped_01.jpg"))

    # Step 2: same finding as a plain photo.
    assert result.status == NO_MANIFEST

    # Step 3: same fully-unsure opinion.
    mass = to_mass(result)
    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)

    # Step 4: even as the driving signal, the verdict abstains.
    verdict = classify(mass, driver="provenance")
    assert verdict == UNKNOWN


# Our placeholder AI image currently carries no manifest either, so the
# honest answer today is "I don't know" (forensics will speak in Step 4).
def test_placeholder_ai_image_has_no_manifest():
    """synthetic_01.png -> no_manifest (placeholder, unsigned)."""
    result = check(fixture_path("synthetic_01.png"))
    assert result.status == NO_MANIFEST
    mass = to_mass(result)
    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)


# The self-signed camera fixture must read as 95% real with no AI claim.
def test_valid_camera_fixture():
    """c2pa_valid_camera.jpg -> valid_camera, 95% real."""
    # Step 1: run the checker.
    result = check(fixture_path("c2pa_valid_camera.jpg"))

    # Step 2: the finding must be camera proof, no AI claim.
    assert result.status == VALID_CAMERA
    assert result.ai_claim is False
    assert result.validation_state == "Valid"

    # Step 3: the opinion must be 95% real.
    mass = to_mass(result)
    assert mass.m_auth == pytest.approx(0.95, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.05, abs=1e-9)

    # Step 4: as the driving signal this is hard proof.
    verdict = classify(mass, driver="provenance")
    assert verdict == PROVENANCE_AVAILABLE


# The self-signed AI fixture must read as 95% AI with the AI claim set.
def test_valid_ai_fixture():
    """c2pa_valid_ai.png -> valid_ai, 95% AI."""
    # Step 1: run the checker.
    result = check(fixture_path("c2pa_valid_ai.png"))

    # Step 2: the finding must be AI proof with the claim flag on.
    assert result.status == VALID_AI
    assert result.ai_claim is True
    assert result.validation_state == "Valid"

    # Step 3: the opinion must be 95% AI.
    mass = to_mass(result)
    assert mass.m_auth == pytest.approx(0.0, abs=1e-9)
    assert mass.m_synth == pytest.approx(0.95, abs=1e-9)
    assert mass.m_uncert == pytest.approx(0.05, abs=1e-9)

    # Step 4: as the driving signal this is hard proof.
    verdict = classify(mass, driver="provenance")
    assert verdict == VERIFIED_AI_ORIGIN


# The mapping table itself, tested without files: tampered leans AI 50/50,
# unreadable is fully unsure, and garbage statuses fail loudly.
def test_to_mass_branches_without_files():
    """to_mass covers tampered / unreadable / unknown-status."""
    # Step 1: a broken file is a hint, not proof.
    tampered = ProvenanceResult(status=TAMPERED, reason="test")
    tampered_mass = to_mass(tampered)
    assert tampered_mass.m_synth == pytest.approx(0.5, abs=1e-9)
    assert tampered_mass.m_uncert == pytest.approx(0.5, abs=1e-9)
    assert tampered_mass.m_auth == pytest.approx(0.0, abs=1e-9)

    # Step 2: an unreadable file is total ignorance.
    unreadable = ProvenanceResult(status=UNREADABLE, reason="test")
    unreadable_mass = to_mass(unreadable)
    assert unreadable_mass.m_uncert == pytest.approx(1.0, abs=1e-9)

    # Step 3: a made-up status is a programmer bug -> loud error.
    bogus = ProvenanceResult(status="definitely-real-trust-me", reason="test")
    with pytest.raises(ValueError):
        to_mass(bogus)


# A nonexistent path must not crash the checker: UNREADABLE, fully unsure.
def test_missing_file_is_unreadable():
    """Bogus path -> unreadable finding, never an exception."""
    result = check(fixture_path("does-not-exist.jpg"))
    assert result.status == UNREADABLE
    mass = to_mass(result)
    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)
