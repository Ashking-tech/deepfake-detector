"""Step 5: run all 3 checkers and mix their answers into one verdict.

Plain story: you give this file one image path. It runs Checker 1
(history file), Checker 2 (hidden code) and Checker 3 (picture
detective) at the same time, mixes their 3 opinions with the Step-1
calculator, and returns one plain answer with the breakdown.

One checker crashing must never kill the answer: any failure becomes
"I don't know" for that pillar only.
"""

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from app.fusion.classify import classify
from app.fusion.dempster import belief_synth, plausibility_synth, sequential_fuse
from app.fusion.mass import MassFunction, vacuous

# Biggest upload we accept (10 MB is plenty for a demo image).
MAX_FILE_MB = 10

# Only these image types are allowed (matches the image-only MVP scope).
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Seconds each checker may take before we give up on it.
PILLAR_TIMEOUT = 30.0


# Checks the file is real, small enough, and actually an image.
# Returns (ok True/False, reason why).
def check_file(path) -> tuple:
    """Validate path: exists, size cap, extension, magic bytes."""
    # Step 1: turn it into a Path and check it exists.
    file_path = Path(path)

    if not file_path.exists():
        return False, "file does not exist"

    if not file_path.is_file():
        return False, "not a regular file"

    # Step 2: check the size cap.
    max_bytes = MAX_FILE_MB * 1024 * 1024

    size = file_path.stat().st_size

    if size > max_bytes:
        return False, f"file too big ({size} bytes, max {max_bytes})"

    if size == 0:
        return False, "file is empty"

    # Step 3: check the extension.
    ext = file_path.suffix.lower()

    if ext not in ALLOWED_EXT:
        return False, f"bad extension '{ext}', allowed: {sorted(ALLOWED_EXT)}"

    # Step 4: check the magic bytes (real file type, not just the name).
    with open(file_path, "rb") as handle:
        head = handle.read(12)

    is_jpeg = head[:2] == b"\xff\xd8"

    is_png = head[:8] == b"\x89PNG\r\n\x1a\n"

    is_riff = head[:4] == b"RIFF"
    is_webp = head[8:12] == b"WEBP"
    is_webp_file = is_riff and is_webp

    if not (is_jpeg or is_png or is_webp_file):
        return False, "bad magic bytes (not a real jpg/png/webp)"

    return True, "ok"


# SHA-256 hash of the file, for logs (proves which file was checked).
def hash_file(path) -> str:
    """Return hex SHA-256 of the file, or 'unreadable' on failure."""
    # Step 1: read the bytes in small chunks (big files stay cheap).
    try:
        hasher = hashlib.sha256()

        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(65536)

                if not chunk:
                    break

                hasher.update(chunk)

        # Step 2: return the hex digest.
        digest = hasher.hexdigest()

        return digest

    except Exception:
        return "unreadable"


# Runs Checker 1 (history file). Never raises: failure -> fully unsure.
def _run_provenance(path) -> tuple:
    """Run provenance check, return (mass, info dict). Never raises."""
    from app.pillars.provenance.verify import check, to_mass

    # Step 1: run the checker.
    try:
        result = check(path)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"provenance crashed: {exc}"}

        return mass, info

    # Step 2: convert the finding into masses.
    try:
        mass = to_mass(result)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"provenance mapping failed: {exc}"}

        return mass, info

    # Step 3: pack the human-readable info for the UI.
    info = {"status": result.status, "reason": result.reason}

    return mass, info


# Runs Checker 2 (hidden code). Never raises: failure -> fully unsure.
def _run_watermark(path) -> tuple:
    """Run watermark detect, return (mass, info dict). Never raises."""
    from app.pillars.watermark.detect import detect, to_mass

    # Step 1: run the checker.
    try:
        result = detect(path)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"watermark crashed: {exc}"}

        return mass, info

    # Step 2: convert the finding into masses.
    try:
        mass = to_mass(result)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"watermark mapping failed: {exc}"}

        return mass, info

    # Step 3: pack the human-readable info for the UI.
    info = {
        "status": "found" if result.found else "absent",
        "reason": result.reason,
        "ber": result.ber,
    }

    return mass, info


# Runs Checker 3 (picture detective). Never raises: failure -> fully unsure.
def _run_forensics(path) -> tuple:
    """Run forensics check, return (mass, info dict). Never raises."""
    from app.pillars.forensics.detect import check, to_mass

    # Step 1: run the checker.
    try:
        result = check(path)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"forensics crashed: {exc}"}

        return mass, info

    # Step 2: convert the finding into masses.
    try:
        mass = to_mass(result)
    except Exception as exc:
        mass = vacuous()

        info = {"status": "error", "reason": f"forensics mapping failed: {exc}"}

        return mass, info

    # Step 3: pack the human-readable info for the UI.
    info = {"status": "scored", "reason": result.reason, "score": result.score}

    return mass, info


# Picks which pillar drove the verdict: the most confident one.
# Confidence = decided points (1 - unsure). All unsure -> "none".
def pick_driver(masses: dict) -> str:
    """Return 'provenance' | 'watermark' | 'forensics' | 'none'."""
    # Step 1: score each pillar by how decided it is.
    best_name = "none"

    best_confident = 0.0

    for name in ["provenance", "watermark", "forensics"]:
        mass = masses[name]

        confident = 1.0 - mass.m_uncert

        if confident > best_confident:
            best_confident = confident
            best_name = name

    # Step 2: nobody decided anything -> no driver.
    if best_confident <= 0.0:
        return "none"

    return best_name


# Main entry: check one image with all 3 checkers and fuse the answer.
# Always returns a dict, never raises (bad input -> Unknown answer).
def verify(path, timeout: float = PILLAR_TIMEOUT) -> dict:
    """Run 3 pillars in parallel, fuse, classify. Returns result dict."""
    # Step 1: start the clock.
    start = time.time()

    # Step 2: validate the file first (cheap, before heavy models).
    ok, reason = check_file(path)

    if not ok:
        fused = vacuous()

        elapsed_ms = (time.time() - start) * 1000.0

        return {
            "status": "Unknown",
            "reason": reason,
            "masses": {
                "m_auth": 0.0,
                "m_synth": 0.0,
                "m_uncert": 1.0,
            },
            "belief": 0.0,
            "plausibility": 1.0,
            "K": 0.0,
            "driver": "none",
            "per_pillar": {},
            "file_hash": hash_file(path),
            "latency_ms": round(elapsed_ms, 1),
        }

    # Step 3: run the 3 checkers at the same time.
    jobs = {
        "provenance": _run_provenance,
        "watermark": _run_watermark,
        "forensics": _run_forensics,
    }

    masses = {}
    per_pillar = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        # Submit all three jobs at once.
        futures = {}

        for name in jobs:
            func = jobs[name]

            future = pool.submit(func, path)

            futures[name] = future

        # Collect each answer (a slow checker times out -> unsure).
        for name in jobs:
            future = futures[name]

            try:
                mass, info = future.result(timeout=timeout)
            except Exception as exc:
                mass = vacuous()

                info = {"status": "timeout", "reason": f"{name} timed out: {exc}"}

            masses[name] = mass

            # Save both the mass numbers and the human reason.
            per_pillar[name] = {
                "m_auth": mass.m_auth,
                "m_synth": mass.m_synth,
                "m_uncert": mass.m_uncert,
                "detail": info,
            }

    # Step 4: mix the 3 opinions with the Step-1 calculator.
    mass_list = [masses["provenance"], masses["watermark"], masses["forensics"]]

    fused, conflict_k = sequential_fuse(mass_list)

    # Step 5: pick the driver (most confident pillar).
    driver = pick_driver(masses)

    # Step 6: pick 1 of the 4 answers.
    verdict = classify(fused, driver=driver, k=conflict_k)

    # Step 7: stop the clock and build the answer dict.
    elapsed_ms = (time.time() - start) * 1000.0

    belief = belief_synth(fused)

    plausibility = plausibility_synth(fused)

    result = {
        "status": verdict,
        "reason": f"driver={driver}",
        "masses": {
            "m_auth": fused.m_auth,
            "m_synth": fused.m_synth,
            "m_uncert": fused.m_uncert,
        },
        "belief": belief,
        "plausibility": plausibility,
        "K": conflict_k,
        "driver": driver,
        "per_pillar": per_pillar,
        "file_hash": hash_file(path),
        "latency_ms": round(elapsed_ms, 1),
    }

    return result


# Appends one JSON line per check to logs/verify.jsonl (audit trail).
def log_result(result: dict, path, log_path: str = "logs/verify.jsonl") -> None:
    """Append result as one JSON line. Never raises (logging is best-effort)."""
    # Step 1: build the log record.
    try:
        record = {}

        record["path"] = str(path)

        record["file_hash"] = result.get("file_hash")

        record["status"] = result.get("status")

        record["driver"] = result.get("driver")

        record["masses"] = result.get("masses")

        record["K"] = result.get("K")

        record["latency_ms"] = result.get("latency_ms")

        line = json.dumps(record)
    except Exception:
        return

    # Step 2: create the folder and append the line.
    try:
        log_file = Path(log_path)

        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a") as handle:
            handle.write(line + "\n")
    except Exception:
        return
