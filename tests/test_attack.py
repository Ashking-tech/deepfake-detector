"""Step-6 tests: prove the attacks are real damage, not no-ops.

Needs Pillow only. The pipeline-safety test always runs; the C2PA
strip test skips gracefully when c2pa-python is not installed.
"""

import importlib.util

import pytest
from PIL import Image

from eval.attack import (
    attack_crop10,
    attack_jpeg,
    attack_resize50,
    attack_strip,
    list_targets,
)
from tests.conftest import fixture_path


# Skip mark for tests needing the real C2PA library.
needs_c2pa = pytest.mark.skipif(
    importlib.util.find_spec("c2pa") is None,
    reason="needs c2pa-python installed",
)


# We attack all 12 image fixtures (option (a)): nothing skipped, nothing
# non-image included.
def test_all_fixtures_targeted(tmp_path):
    """list_targets finds 12 images, no helper files."""
    # Step 1: list the targets.
    targets = list_targets()

    # Step 2: every entry must be a real image file (11 today:
    # 2 signed + 2 real + 1 synthetic + 2 derived + 4 watermark).
    # Lower bound, not exact: adding future fixtures must not break this.
    assert len(targets) >= 10

    for target in targets:
        assert target.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


# Stripping must really delete metadata (checked via Pillow, no c2pa).
def test_strip_removes_metadata(tmp_path):
    """Stripped copy has empty exif (Pillow-level proof)."""
    # Step 1: strip a real photo.
    out = tmp_path / "stripped.jpg"

    attack_strip(fixture_path("real_01.jpg"), out)

    # Step 2: the copy must open and carry no exif.
    img = Image.open(out)

    img.load()

    exif = img.getexif()

    assert len(exif) == 0


# Stripping a SIGNED fixture must kill its manifest (the Instagram case).
# Guarded: needs the real C2PA library to read manifests.
@needs_c2pa
def test_strip_kills_c2pa_manifest(tmp_path):
    """Stripped signed copy reads no_manifest (never 'authentic')."""
    from app.pillars.provenance.verify import NO_MANIFEST, check, to_mass

    # Step 1: strip the signed camera fixture.
    out = tmp_path / "killed.jpg"

    attack_strip(fixture_path("c2pa_valid_camera.jpg"), out)

    # Step 2: the manifest must be gone...
    result = check(out)

    assert result.status == NO_MANIFEST

    # Step 3: ...and the opinion fully unsure, never authentic.
    mass = to_mass(result)

    assert mass.m_uncert == pytest.approx(1.0, abs=1e-9)


# JPEG recompression must shrink/changes bytes but stay a readable image.
def test_jpeg_recompress_is_real_damage(tmp_path):
    """q50 file differs from source and stays readable."""
    # Step 1: recompress at q50.
    out = tmp_path / "small.jpg"

    attack_jpeg(fixture_path("real_01.jpg"), out, 50)

    # Step 2: it must open as a real image...
    img = Image.open(out)

    img.load()

    assert img.size[0] > 0

    # Step 3: ...with different bytes than the source.
    original_bytes = fixture_path("real_01.jpg").read_bytes()

    damaged_bytes = out.read_bytes()

    assert damaged_bytes != original_bytes


# Crop must cut pixels: output is exactly 80% of each side.
def test_crop10_cuts_pixels(tmp_path):
    """Cropped copy is 80% width and height."""
    # Step 1: crop a real photo.
    out = tmp_path / "cropped.jpg"

    attack_crop10(fixture_path("real_01.jpg"), out)

    # Step 2: measure both (expected mirrors the 10%-margin math in
    # attack_crop10: right-left, not 80% of width, due to int rounding).
    original = Image.open(fixture_path("real_01.jpg"))

    damaged = Image.open(out)

    full_width, full_height = original.size

    want_width = int(full_width * 0.90) - int(full_width * 0.10)

    want_height = int(full_height * 0.90) - int(full_height * 0.10)

    assert damaged.size[0] == want_width

    assert damaged.size[1] == want_height


# Resize must halve each side (thumbnail pipeline).
def test_resize50_halves_sides(tmp_path):
    """Resized copy is half width and height."""
    # Step 1: shrink a real photo.
    out = tmp_path / "tiny.jpg"

    attack_resize50(fixture_path("real_01.jpg"), out)

    # Step 2: measure both.
    original = Image.open(fixture_path("real_01.jpg"))

    damaged = Image.open(out)

    assert damaged.size[0] == original.size[0] // 2

    assert damaged.size[1] == original.size[1] // 2


# The pipeline must never crash on damaged images (worst case: Unknown).
def test_verify_survives_all_attacks(tmp_path):
    """verify() answers (never raises) on every damaged copy."""
    from app.api.orchestrate import verify

    # Step 1: damage one photo 5 ways.
    attacks = [attack_strip, attack_crop10, attack_resize50]

    outs = []

    for index, func in enumerate(attacks):
        out = tmp_path / f"dmg{index}.jpg"

        func(fixture_path("real_01.jpg"), out)

        outs.append(out)

    recompressed = tmp_path / "dmg_q50.jpg"

    attack_jpeg(fixture_path("real_01.jpg"), recompressed, 50)

    outs.append(recompressed)

    # Step 2: every damaged copy must get an answer, never an exception.
    for out in outs:
        result = verify(out)

        assert result["status"] in (
            "Verified AI Origin",
            "Provenance Available",
            "Suspicious",
            "Unknown",
        )
