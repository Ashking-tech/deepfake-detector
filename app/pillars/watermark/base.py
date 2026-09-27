"""Checker 2 shared types: result format, detector interface, registry.

Plain story: a watermark detector looks for OUR secret 64-bit pattern in an
image's pixels. Different algorithms (dwtDct today, RivaGAN later) all speak
through the same WatermarkDetector interface, and the registry maps a plain
name ("dwtDct") to the class that implements it. Swapping algorithms later
means changing one registration line, never the fusion code.

Honest limitation (do not remove): blind detection needs the EXPECTED
payload to compare against. This pillar detects watermarks WE embedded
(like SynthID detects with its own key). It cannot read third-party
watermarks (SynthID, InvisMark, Stable Signature) without their keys.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WatermarkResult:
    """Plain finding from one watermark detector.

    found: True when the decoded bits match our magic closely enough.
    ber: bit error rate in [0,1] — fraction of the 64 bits that mismatched.
        0.0 = perfect match. ~0.5 = random garbage (nothing there).
    reason: one-line human explanation, goes straight into logs and UI.
    detector: name of the algorithm that produced this finding.
    """

    found: bool
    ber: float
    reason: str
    detector: str = "unknown"


# Common interface every watermark algorithm implements.
class WatermarkDetector(ABC):
    """One watermark algorithm (dwtDct, rivaGan, ...).

    Subclasses set `name` and `payload_bits`, then implement detect().
    """

    # Short name used in logs, tests, and the registry (e.g. "dwtDct").
    name: str = "?"

    # How many bits this algorithm reads per image (64 for dwtDct,
    # 32 for RivaGAN — each algorithm defines its own magic length).
    payload_bits: int = 0

    # Reads the image at path, decodes bits, compares to our magic.
    # Returns a WatermarkResult (implementations should not raise on
    # ordinary bad input; detect() in detect.py also guards them).
    @abstractmethod
    def detect(self, path) -> WatermarkResult:
        """Decode the image at path and compare to our magic payload."""
        raise NotImplementedError


# Maps detector names to factory functions. Factories (not instances) are
# stored so heavy libraries import lazily, only when actually used.
DETECTORS: dict = {}


# Registers one detector factory under a short name.
def register_detector(name: str, factory):
    """Register factory() -> WatermarkDetector under `name`."""
    DETECTORS[name] = factory


# Looks up a detector by name and builds it.
def get_detector(name: str) -> WatermarkDetector:
    """Build the detector registered as `name`. Raises KeyError naming the
    known detectors when the name is unknown (fail loudly, fail early)."""
    if name not in DETECTORS:
        known = sorted(DETECTORS.keys())
        raise KeyError(f"unknown watermark detector '{name}', known: {known}")
    factory = DETECTORS[name]
    detector = factory()
    return detector
