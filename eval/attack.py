"""Step 6: play attacker on our own fixtures, then measure survival.

Plain story: attackers don't submit original files — they strip
metadata, recompress, crop and resize. This script does those 5 things
to every fixture, runs the full 3-checker pipeline on each damaged
copy, and writes eval/report.md saying what survived.

Needs Pillow only (no torch, no c2pa, no opencv). Run it with:
    python eval/attack.py
"""

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `app` importable when run as `python eval/attack.py` (script dir
# lands on sys.path, not the project root).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

# Folder holding the original test images.
FIXTURES = Path("tests/fixtures")

# Folder where damaged copies land (regenerable, gitignored).
ATTACKED = Path("eval/attacked")

# The report this script writes.
REPORT = Path("eval/report.md")

# Image files we attack (everything viewable, skip helper files).
ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}

# Files that are not images, never attack these.
SKIP_NAMES = {".gitkeep", "SOURCES.md"}


# True when an optional heavy library can be imported.
def _have(name: str) -> bool:
    """Check a module is importable without importing it."""
    found = importlib.util.find_spec(name) is not None

    return found


# Lists every attackable fixture, sorted by name (stable report order).
def list_targets() -> list:
    """Return sorted Paths of fixtures with image suffixes."""
    targets = []

    for child in sorted(FIXTURES.iterdir()):
        if child.name in SKIP_NAMES:
            continue

        if child.suffix.lower() not in ALLOWED_SUFFIX:
            continue

        if not child.is_file():
            continue

        targets.append(child)

    return targets


# Attack 1: delete all metadata (simulates Instagram/Facebook upload).
# Pillow drops exif unless we pass it, so a plain re-save strips it.
def attack_strip(src: Path, dst: Path) -> None:
    """Re-save with zero metadata (no exif, no icc profile)."""
    # Step 1: open the original.
    img = Image.open(src)

    # Step 2: flatten palettes to plain RGB/RGBA (keeps PNG valid).
    flat = img.convert("RGB") if img.mode == "P" else img

    # Step 3: save with no exif and no icc profile attached.
    flat.save(dst)


# Attack 2+3: recompress as JPEG (simulates WhatsApp forwarding).
def attack_jpeg(src: Path, dst: Path, quality: int) -> None:
    """Re-save as JPEG at the given quality (drops all metadata too)."""
    # Step 1: open and force RGB (JPEG has no alpha channel).
    img = Image.open(src)

    rgb = img.convert("RGB")

    # Step 2: save at the requested quality, no metadata.
    rgb.save(dst, "JPEG", quality=quality)


# Attack 4: cut 10% off every edge (simulates profile-picture crops).
def attack_crop10(src: Path, dst: Path) -> None:
    """Center-crop to 80% width and height, keep original format."""
    # Step 1: open and measure.
    img = Image.open(src)

    width, height = img.size

    # Step 2: compute the center 80% box (10% margin each side).
    left = int(width * 0.10)
    top = int(height * 0.10)
    right = int(width * 0.90)
    bottom = int(height * 0.90)

    # Step 3: crop and save.
    cropped = img.crop((left, top, right, bottom))

    cropped.save(dst)


# Attack 5: shrink to half size (simulates thumbnail pipelines).
def attack_resize50(src: Path, dst: Path) -> None:
    """Downscale to 50% with bilinear filtering, keep original format."""
    # Step 1: open and measure.
    img = Image.open(src)

    width, height = img.size

    # Step 2: halve each side (never below 1 pixel).
    small_width = max(1, width // 2)
    small_height = max(1, height // 2)

    # Step 3: resize and save.
    small = img.resize((small_width, small_height), Image.BILINEAR)

    small.save(dst)


# The 5 attacks: name -> function(src, dst). Order is report order.
ATTACKS = {
    "strip": attack_strip,
    "jpg70": lambda s, d: attack_jpeg(s, d, 70),
    "jpg50": lambda s, d: attack_jpeg(s, d, 50),
    "crop10": attack_crop10,
    "resize50": attack_resize50,
}


# Runs the full pipeline on one file, never raises (broken file -> Unknown).
def check_one(path: Path) -> dict:
    """Run verify() on path, return the result dict (never raises)."""
    from app.api.orchestrate import verify

    # Step 1: run the pipeline (it never raises by contract).
    try:
        result = verify(path)
    except Exception as exc:
        result = {}

        result["status"] = "Unknown"

        result["reason"] = f"verify crashed (should never happen): {exc}"

        result["driver"] = "none"

        result["masses"] = {"m_auth": 0.0, "m_synth": 0.0, "m_uncert": 1.0}

        result["K"] = 0.0

    return result


# One-line verdict summary for the report table.
def short_verdict(result: dict) -> str:
    """Format 'Status (driver)' for a result dict."""
    status = result.get("status", "Unknown")

    driver = result.get("driver", "none")

    text = f"{status} ({driver})"

    return text


# Builds the whole report: attack every fixture, check every copy.
def main() -> None:
    """Attack all fixtures, verify all copies, write eval/report.md."""
    # Step 1: make the output folder.
    ATTACKED.mkdir(parents=True, exist_ok=True)

    # Step 2: find the targets.
    targets = list_targets()

    print(f"targets: {len(targets)}, attacks: {len(ATTACKS)}")

    # Step 3: attack every target, check original + every copy.
    rows = []

    for target in targets:
        # Original first (the baseline to compare against).
        base = check_one(target)

        rows.append((target.name, "original", short_verdict(base), base))

        print(f"  {target.name} original -> {short_verdict(base)}")

        # Then each attack.
        for attack_name in ATTACKS:
            func = ATTACKS[attack_name]

            out_name = f"{target.stem}__{attack_name}{target.suffix}"

            # Quality attacks always land as .jpg (they ARE recompression).
            if attack_name.startswith("jpg"):
                out_name = f"{target.stem}__{attack_name}.jpg"

            out_path = ATTACKED / out_name

            try:
                func(target, out_path)

                damaged = check_one(out_path)

                verdict = short_verdict(damaged)
            except Exception as exc:
                verdict = f"ATTACK FAILED: {exc}"

                damaged = {}

            rows.append((target.name, attack_name, verdict, damaged))

            print(f"  {target.name} {attack_name} -> {verdict}")

    # Step 4: count how often the verdict survived the attack.
    kept = 0

    total = 0

    unknowns = 0

    for target_name, attack_name, verdict, damaged in rows:
        if attack_name == "original":
            continue

        total += 1

        # Find this target's baseline verdict.
        baseline = ""

        for other_name, other_attack, other_verdict, other in rows:
            if other_name == target_name and other_attack == "original":
                baseline = other_verdict

                break

        if verdict == baseline:
            kept += 1

        if verdict.startswith("Unknown"):
            unknowns += 1

    # Step 5: write the markdown report.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []

    lines.append("# Step 6 attack-lite report")

    lines.append("")

    lines.append(f"Generated: {now}")

    lines.append("")

    lines.append("Environment (pillars missing a library vote unsure):")

    lines.append("")

    lines.append(f"- c2pa (Checker 1): {'present' if _have('c2pa') else 'MISSING'}")

    has_wm = _have('imwatermark') and _have('onnxruntime') and _have('cv2')

    lines.append(f"- watermark stack (Checker 2): {'present' if has_wm else 'MISSING'}")

    lines.append(f"- torch backbones (Checker 3 deep): {'present' if _have('torch') else 'MISSING (FFT heuristic only)'}")

    lines.append("")

    lines.append(f"Summary: verdict kept after attack {kept}/{total}, honest Unknown {unknowns}/{total}.")

    lines.append("")

    lines.append("| Image | Attack | Verdict (driver) |")

    lines.append("|---|---|---|")

    for target_name, attack_name, verdict, damaged in rows:
        lines.append(f"| {target_name} | {attack_name} | {verdict} |")

    lines.append("")

    lines.append("Reading guide: same verdict as `original` = pillar survived.")

    lines.append("`Unknown` after attack = honest abstain (attack worked, we admit it).")

    lines.append("A flip (e.g. Suspicious -> Unknown) is graceful damage, not a lie.")

    REPORT.write_text("\n".join(lines) + "\n")

    print(f"wrote {REPORT} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
