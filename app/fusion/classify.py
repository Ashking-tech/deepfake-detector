"""Verdict: turns fused numbers + who-drove-them into 1 of 4 plain answers.

Why the driver matters: 79% fake driven by a cryptographic history file is
proof ("Verified AI Origin"), while the same 79% driven only by the picture
detective is a strong hint but not proof ("Suspicious"). The driver is the
pillar with the largest confident (non-unsure) contribution; callers pass it
in as a plain word.
"""

from .mass import MassFunction

VERIFIED_AI_ORIGIN = "Verified AI Origin"
PROVENANCE_AVAILABLE = "Provenance Available"
SUSPICIOUS = "Suspicious"
UNKNOWN = "Unknown"

# Above this conflict the checkers contradict each other so badly that any
# binary answer would be a coin flip -> force Unknown (honest abstain).
HIGH_CONFLICT_K = 0.6

# Fused FAKE share needed for a confident answer (with a trusted driver).
STRONG_SYNTH = 0.7

# Fused REAL share needed for a confident camera answer (provenance only).
STRONG_AUTH = 0.7

# Above this unsure share, we admit ignorance instead of guessing.
HIGH_UNCERT = 0.5

# Fused FAKE share needed for a hint-level answer (no proof required).
LEAN_SYNTH = 0.5


# Picks the final human-readable answer from the fused opinion.
# driver: "provenance" | "watermark" | "forensics" | "none" — which checker
# contributed the most confident points (caller decides, we just read it).
# k: conflict level of the last mix (from sequential_fuse).
# Order: contradiction first (safety), then crypto proofs, then hints.
def classify(
    fused: MassFunction,
    driver: str = "none",
    k: float = 0.0,
) -> str:
    """Map fused masses to one of the 4 statuses.

    Never raises; Unknown is the safe default.
    """
    # Safety gate 1: checkers flatly contradict each other.
    high_conflict = k > HIGH_CONFLICT_K
    if high_conflict:
        # Refuse to guess.
        return UNKNOWN

    # Proof gate 1: hard proof (history file or hidden code) says AI-made.
    strong_fake = fused.m_synth > STRONG_SYNTH
    trusted_fake_driver = driver == "provenance" or driver == "watermark"
    if strong_fake and trusted_fake_driver:
        return VERIFIED_AI_ORIGIN

    # Proof gate 2: hard proof (history file) says camera-made, no AI edits.
    strong_real = fused.m_auth > STRONG_AUTH
    trusted_real_driver = driver == "provenance"
    if strong_real and trusted_real_driver:
        return PROVENANCE_AVAILABLE

    # Honesty gate: mostly shrugs, so say so.
    mostly_unsure = fused.m_uncert > HIGH_UNCERT
    if mostly_unsure:
        return UNKNOWN

    # Hint gate: no proof, but the picture detective smells AI artifacts.
    lean_fake = fused.m_synth > LEAN_SYNTH
    if lean_fake:
        return SUSPICIOUS

    # Nothing fired -> safest answer.
    return UNKNOWN
