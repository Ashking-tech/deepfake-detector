"""Checker 3: picture detective (forensic verdict mapping).

Plain story: this checker looks at the picture ITSELF — no history file,
no hidden code needed. It is the last safety net when attackers strip
everything else. In Step 4 the live signal is the FFT spectral-slope
heuristic; the two deep backbones run and log their vectors but vote
with ZERO weight until Phase 7 trains their heads on real datasets.

Because the live signal is an uncalibrated heuristic, this pillar is
capped at "Suspicious" by construction: BASE_UNCERTAINTY is high enough
that even a fully fake-looking score (1.0) yields m_synth = 0.60 at
most — above the 0.5 Suspicious line, below the 0.7 proof line. The
pillar can hint, never prove. That cap lifts only when dataset
calibration lands in Phase 7.
"""

from app.fusion.mass import MassFunction

from . import fft_branch, preprocess

# Uncertainty floor for the heuristic signal: with score=1.0 this gives
# m_synth = 0.60 (Suspicious, never proof). Documented cap, not tuning.
BASE_UNCERTAINTY = 0.40

# Extra uncertainty added when the slope sits in the doubtful middle.
AMBIGUOUS_PENALTY = 0.20

# Hard ceiling so uncertainty never eats the whole opinion.
MAX_UNCERTAINTY = 0.80

# Temperature for the slope->score squash (1.0 = uncalibrated identity).
# Phase 7 fits this on real datasets; the hook is ready, not tuned.
TEMPERATURE = 1.0

# Deep-backbone vote weights (0.0 = wired but silent until Phase 7).
CONVNEXT_WEIGHT = 0.0
DINO_WEIGHT = 0.0


# Plain finding from the picture detective.
class ForensicsResult:
    """score: P(image is AI-made) in [0,1] (heuristic, uncalibrated).
    uncertainty: honest unsure share in [0,1].
    reason: one-line human explanation for logs and UI.
    details: measured numbers (slope, embeddings shapes) for inspection.
    """

    def __init__(self, score, uncertainty, reason, details=None):
        self.score = float(score)
        self.uncertainty = float(uncertainty)
        self.reason = reason
        if details is None:
            details = {}
        self.details = details


# Runs the full detective pipeline on one image file. Never raises: any
# failure becomes a high-uncertainty finding (a broken checker must never
# crash the pipeline or fake a verdict).
def check(path):
    """Analyze path. Returns ForensicsResult (never raises)."""
    # Step 1: load pixels (None = unreadable file).
    image = preprocess.load_image(path)
    if image is None:
        return ForensicsResult(
            score=0.5,
            uncertainty=1.0,
            reason=f"could not read image: {path}",
            details={},
        )

    # Step 2: shared preparation (square + 224x224).
    small = preprocess.prepare(image)

    # Step 3: FFT heuristic (THE live signal in Step 4).
    rgb = preprocess.to_array(small)
    gray = rgb.mean(axis=2)
    slope = fft_branch.spectral_slope(gray)
    score = fft_branch.slope_to_score(slope, temperature=TEMPERATURE)

    # Step 4: uncertainty starts at the floor.
    uncertainty = BASE_UNCERTAINTY

    # Step 5: doubtful-middle slopes earn extra uncertainty.
    if fft_branch.is_ambiguous(slope):
        uncertainty = uncertainty + AMBIGUOUS_PENALTY

    # Step 6: hard ceiling.
    if uncertainty > MAX_UNCERTAINTY:
        uncertainty = MAX_UNCERTAINTY

    # Step 7: deep backbones run for inspection, vote nothing (weight 0).
    details = {}
    details["slope"] = slope
    details["convnext_weight"] = CONVNEXT_WEIGHT
    details["dino_weight"] = DINO_WEIGHT
    try:
        from . import convnext, dino_probe

        conv_vector = convnext.embed(rgb)
        details["convnext_dim"] = len(conv_vector)
        cls_vector = dino_probe.cls_embed(small)
        details["dino_dim"] = len(cls_vector)
        probe_vote, trained = dino_probe.probe_score(cls_vector)
        details["dino_probe_vote"] = probe_vote
        details["dino_probe_trained"] = trained
    except Exception as exc:
        details["backbone_error"] = str(exc)[:120]

    # Step 8: plain-language reason naming the driver.
    reason = f"spectral slope {slope:.2f} -> P(fake) {score:.2f} (heuristic)"
    result = ForensicsResult(
        score=score,
        uncertainty=uncertainty,
        reason=reason,
        details=details,
    )
    return result


# Converts a finding into the 3-number opinion the mixer understands.
def to_mass(result: ForensicsResult) -> MassFunction:
    """Map score + uncertainty to masses. Guards ranges loudly."""
    # Step 1: validate inputs (programmer bugs fail here, not silently).
    if result.score < 0.0 or result.score > 1.0:
        raise ValueError(f"score out of range [0,1]: {result.score}")
    if result.uncertainty < 0.0 or result.uncertainty > 1.0:
        raise ValueError(f"uncertainty out of range [0,1]: {result.uncertainty}")

    # Step 2: confident share splits by score...
    confident = 1.0 - result.uncertainty
    synth_share = result.score * confident
    auth_share = (1.0 - result.score) * confident

    # Step 3: ...and the rest stays honestly unsure.
    mass = MassFunction(
        m_auth=auth_share,
        m_synth=synth_share,
        m_uncert=result.uncertainty,
    )
    return mass
