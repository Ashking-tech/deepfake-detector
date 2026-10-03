"""FFT branch: frequency-fingerprint heuristic (the live Step-4 signal).

Plain story: natural camera photos concentrate their energy in LOW
frequencies (big smooth shapes), with energy falling off toward high
frequencies (fine grain). Our procedural synthetic placeholder does the
opposite — checkerboard + noise spreads energy FLAT across all bands.
The "spectral slope" (low-band energy minus high-band energy) captures
this in one number: ~7 for our real photos, ~1.2 for the placeholder.

HONESTY WARNING (do not delete): this separates OUR fixtures, not real
AI images from real photos. Real diffusion outputs need dataset
calibration (FaceForensics++/Celeb-DF) in Phase 7, which will re-fit
SLOPE_THRESHOLD and SLOPE_WIDTH below. Until then this branch is a
fixture-calibrated placeholder that can only ever vote "Suspicious"
(never proof) — see detect.py's uncertainty floor.

Measured 2026-09 on tests/fixtures through the detect.py pipeline
(center-crop, 224px, RGB-mean gray):
  real_01.jpg       slope 7.97  -> P(fake) 0.02
  real_02.png       slope 7.33  -> P(fake) 0.03
  synthetic_01.png  slope 1.58  -> P(fake) 0.92
JPEG q60 and rotation change the slope by < 0.1 (robust to re-encode).
"""

import numpy as np

# Radius fractions splitting the spectrum into bands (of half-size).
# Low band = big shapes, high band = fine grain. Middle bands unused.
LOW_RADIUS = 0.2
HIGH_LOW_RADIUS = 0.8

# Slope value halfway between our real photos (~7) and our placeholder
# (~1.2). A slope below this leans FAKE, above it leans REAL.
SLOPE_THRESHOLD = 4.0

# How wide the unsure zone is around the threshold. Wider = more honest
# about the uncalibrated middle. Re-fit in Phase 7 on real datasets.
SLOPE_WIDTH = 1.0

# Slope zone where even the direction is doubtful: push extra uncertainty.
AMBIGUOUS_LOW = 2.5
AMBIGUOUS_HIGH = 5.5


# Measures mean log-power in the low and high frequency bands.
def band_energies(gray224):
    """Mean log-power of low/high bands. Input: 224x224 float array."""
    # Step 1: 2D Fourier transform, centered (zero frequency in middle).
    spectrum = np.fft.fftshift(np.fft.fft2(gray224))

    # Step 2: log-power so huge DC values don't drown everything.
    power = np.log1p(np.abs(spectrum) ** 2)

    # Step 3: distance of each pixel from the center, as a fraction.
    size = power.shape[0]
    center = size // 2
    rows, cols = np.mgrid[:size, :size]
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    dist = dist / (size / 2)

    # Step 4: average power inside each band mask.
    low_mask = dist < LOW_RADIUS
    high_mask = dist >= HIGH_LOW_RADIUS
    low_energy = power[low_mask].mean()
    high_energy = power[high_mask].mean()

    return float(low_energy), float(high_energy)


# Collapses the two band energies into one slope number.
def spectral_slope(gray224):
    """Low-band energy minus high-band energy. Higher = more natural."""
    # Step 1: measure both bands.
    low_energy, high_energy = band_energies(gray224)

    # Step 2: the gap between them is the slope.
    slope = low_energy - high_energy
    return slope


# Turns a slope into P(image is AI-made) via a logistic curve.
def slope_to_score(slope, temperature=1.0):
    """Map slope to P(fake) in [0,1]. Below threshold -> fake-leaning.

    temperature=1.0 until Phase 7 calibration fits it on real datasets
    (the parameter exists so the hook is ready, not because it is tuned).
    """
    # Step 1: signed distance from the threshold (negative = fake side).
    distance = SLOPE_THRESHOLD - slope

    # Step 2: scale by width and temperature.
    logit = distance / (SLOPE_WIDTH * temperature)

    # Step 3: squash to a probability.
    score = 1.0 / (1.0 + np.exp(-logit))
    return float(score)


# True when the slope sits in the doubtful middle (direction untrusted).
def is_ambiguous(slope):
    """Slope inside (2.5, 5.5) -> ambiguous, deserves extra uncertainty."""
    below_top = slope < AMBIGUOUS_HIGH
    above_bottom = slope > AMBIGUOUS_LOW
    return bool(below_top and above_bottom)
