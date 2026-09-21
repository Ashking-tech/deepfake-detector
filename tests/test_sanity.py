"""Phase 0 sanity: skeleton imports + fixture presence/decodability."""
from pathlib import Path

from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


def test_package_imports():
    import app  # noqa: F401
    import app.api  # noqa: F401
    import app.fusion  # noqa: F401
    import app.pillars.forensics  # noqa: F401
    import app.pillars.provenance  # noqa: F401
    import app.pillars.watermark  # noqa: F401


def test_fixtures_present_and_decodable():
    expected = [
        "real_01.jpg",
        "real_02.png",
        "synthetic_01.png",
        "stripped_01.jpg",
        "recompressed_01.jpg",
    ]
    for name in expected:
        p = FIXTURES / name
        assert p.exists(), f"missing fixture {name}"
        assert p.stat().st_size > 0, f"empty fixture {name}"
        with Image.open(p) as im:
            im.load()
            assert im.size[0] > 0 and im.size[1] > 0
