"""Step-5 tests: prove the demo runner mixes 3 checkers into 1 answer.

No server is started here: every test calls verify() directly.
Gradio is never imported, so these run before `pip install gradio`.
"""

import importlib.util

import pytest

from app.api.orchestrate import check_file, pick_driver, verify
from app.fusion.mass import MassFunction
from tests.conftest import fixture_path


# True when the watermark stack is importable (needed for stamped tests).
def _watermark_ready():
    """Check invisible-watermark + onnxruntime + cv2 without importing."""
    has_lib = importlib.util.find_spec("imwatermark") is not None
    has_onnx = importlib.util.find_spec("onnxruntime") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None

    if has_lib and has_onnx and has_cv2:
        return True

    return False


# Skip mark for the stamped-image test (needs lib + generated fixtures).
needs_watermark = pytest.mark.skipif(
    not _watermark_ready() or not fixture_path("wm_ai_01.jpg").exists(),
    reason="needs watermark lib + stamped fixtures",
)


# Skip mark for the C2PA proof tests (needs c2pa-python installed).
needs_c2pa = pytest.mark.skipif(
    importlib.util.find_spec("c2pa") is None,
    reason="needs c2pa-python installed",
)


# A normal photo with no proof anywhere must answer Unknown (honest).
def test_plain_photo_is_unknown():
    """real_01.jpg -> Unknown (no pillar has proof)."""
    # Step 1: run the full pipeline.
    result = verify(fixture_path("real_01.jpg"))

    # Step 2: the verdict must abstain.
    # NOTE: fused m_uncert is ~0.38 here, not > 0.5 — the forensics FFT
    # leans real on natural photos, so "unsure" does not dominate.
    # Unknown still wins because no gate (proof / hint) fires.
    assert result["status"] == "Unknown"


# Our placeholder AI image has forensics signal but no crypto proof,
# so the answer is Suspicious (hint, never proof).
def test_placeholder_ai_is_suspicious():
    """synthetic_01.png -> Suspicious driven by forensics."""
    # Step 1: run the full pipeline.
    result = verify(fixture_path("synthetic_01.png"))

    # Step 2: hint-level verdict, forensics drove it.
    assert result["status"] == "Suspicious"

    assert result["driver"] == "forensics"


# A valid AI history file is hard proof: Verified AI Origin.
# Guarded: without c2pa-python the pillar safely returns unsure.
@needs_c2pa
def test_c2pa_ai_fixture_is_verified():
    """c2pa_valid_ai.png -> Verified AI Origin, driver provenance."""
    # Step 1: run the full pipeline.
    result = verify(fixture_path("c2pa_valid_ai.png"))

    # Step 2: proof-level verdict with the right driver.
    assert result["status"] == "Verified AI Origin"

    assert result["driver"] == "provenance"


# A valid camera history file is hard proof of the other direction.
# Guarded: without c2pa-python the pillar safely returns unsure.
@needs_c2pa
def test_c2pa_camera_fixture_has_proof():
    """c2pa_valid_camera.jpg -> Provenance Available, driver provenance."""
    # Step 1: run the full pipeline.
    result = verify(fixture_path("c2pa_valid_camera.jpg"))

    # Step 2: proof-level verdict with the right driver.
    assert result["status"] == "Provenance Available"

    assert result["driver"] == "provenance"


# Our stamped image is watermark proof: Verified AI Origin.
# Guarded: needs the watermark library + generated fixtures on disk.
@needs_watermark
def test_stamped_fixture_is_verified():
    """wm_ai_01.jpg -> Verified AI Origin (watermark or provenance driver)."""
    # Step 1: run the full pipeline.
    result = verify(fixture_path("wm_ai_01.jpg"))

    # Step 2: proof-level verdict.
    assert result["status"] == "Verified AI Origin"

    # Step 3: the driver must be a proof pillar (never forensics).
    assert result["driver"] in ("watermark", "provenance")


# A missing file must answer Unknown, never crash.
def test_missing_file_is_unknown():
    """Bogus path -> Unknown with a reason, all unsure."""
    # Step 1: run the pipeline on a path that does not exist.
    result = verify(fixture_path("does-not-exist.jpg"))

    # Step 2: honest abstain, fully unsure.
    assert result["status"] == "Unknown"

    assert result["masses"]["m_uncert"] == pytest.approx(1.0, abs=1e-9)


# A non-image file must be rejected by the file check, before any model.
def test_bad_extension_rejected():
    """SOURCES.md -> Unknown, rejected before checkers run."""
    # Step 1: hand a text file to the validator.
    ok, reason = check_file(fixture_path("SOURCES.md"))

    # Step 2: it must be refused.
    assert ok is False

    # Step 3: the full pipeline must also abstain, not crash.
    result = verify(fixture_path("SOURCES.md"))

    assert result["status"] == "Unknown"


# If one pillar crashes, the other two must still produce an answer.
def test_one_pillar_crash_still_answers(monkeypatch):
    """Broken watermark import -> still Suspicious via forensics."""
    # Step 1: break the watermark pillar only.
    import app.pillars.watermark.detect as watermark_detect

    def broken(path, detector_name="rivaGan"):
        raise RuntimeError("simulated watermark outage")

    monkeypatch.setattr(watermark_detect, "detect", broken)

    # Step 2: the pipeline must still answer from the survivors.
    result = verify(fixture_path("synthetic_01.png"))

    # Step 3: forensics alone still hints fake.
    assert result["status"] == "Suspicious"

    assert result["driver"] == "forensics"


# Driver picking: most decided pillar wins; total shrugs -> "none".
def test_pick_driver_branches():
    """pick_driver prefers decided pillar, 'none' when all unsure."""
    # Step 1: forensics is the most decided here.
    masses = {}

    masses["provenance"] = MassFunction(0.0, 0.0, 1.0)

    masses["watermark"] = MassFunction(0.05, 0.0, 0.95)

    masses["forensics"] = MassFunction(0.10, 0.80, 0.10)

    assert pick_driver(masses) == "forensics"

    # Step 2: nobody decided anything -> no driver.
    unsure = {}

    unsure["provenance"] = MassFunction(0.0, 0.0, 1.0)

    unsure["watermark"] = MassFunction(0.0, 0.0, 1.0)

    unsure["forensics"] = MassFunction(0.0, 0.0, 1.0)

    assert pick_driver(unsure) == "none"


# Every answer must carry the full contract the UI needs.
def test_result_shape():
    """verify() dict has status, masses, K, driver, latency."""
    # Step 1: run on a normal photo.
    result = verify(fixture_path("real_01.jpg"))

    # Step 2: check each required key exists.
    for key in ["status", "masses", "belief", "plausibility", "K"]:
        assert key in result

    for key in ["driver", "per_pillar", "latency_ms", "file_hash"]:
        assert key in result

    # Step 3: all three pillars must have reported.
    assert set(result["per_pillar"].keys()) == {"provenance", "watermark", "forensics"}
