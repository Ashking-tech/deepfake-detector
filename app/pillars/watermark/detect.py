"""Checker 2: hidden code (watermark) protocol + verdict mapping.

Plain story: WE stamp our own images with a secret 64-bit pattern
("DEPAUTH1"). Later, the checker decodes any image and measures how close
the decoded bits are to that pattern. Close = our stamp survived. Far =
nothing stamped here. This mirrors how SynthID works (shared secret key),
except our key is our own published test magic.

What this CANNOT do (honesty clause): read anyone else's watermarks.
SynthID, InvisMark, and Stable Signature each need their own secret keys,
which we don't have. Absence of OUR code means "not stamped by us",
never "therefore real" — hence the weak 5%-real lean below.
"""

from app.fusion.mass import MassFunction, discount

from .base import (
    WatermarkResult,
    get_detector,
    register_detector,
)

# Our magic, as 8 ASCII characters = exactly 64 bits (dwtDct's payload).
# Public on purpose: security-through-obscurity is not the goal here;
# the statistical test (1/2^64 chance of random match) is.
PAYLOAD_ASCII = "DEPAUTH1"
PAYLOAD_BITS = 64

# RivaGAN's separate 32-bit magic (its only supported length): the first
# 4 bytes of our magic. Used only by the Phase 3.5 stub for now.
RIVAGAN_MAGIC = "DEPA"

# Mass shares for a perfectly decoded stamp (matches README's "verified").
VERIFIED_SYNTH = 0.90
VERIFIED_UNCERT = 0.10

# Mass shares when no stamp is found: weak real lean, mostly unsure.
ABSENT_AUTH = 0.05
ABSENT_UNCERT = 0.95


# Turns "DEPAUTH1" into 64 bits, most-significant bit first per byte.
def payload_bits() -> list:
    """Encode PAYLOAD_ASCII as a list of 64 ints (0/1)."""
    bits = []
    for char in PAYLOAD_ASCII:
        code = ord(char)
        binary = format(code, "08b")
        for digit in binary:
            bits.append(int(digit))
    return bits


# Lazy factory: imports the heavy library only when actually built, so
# `import detect` stays cheap and tests can guard on availability.
def _make_dwt():
    """Build the default dwtDct detector (imports imwatermark now)."""
    from .dct_detector import DctDetector

    expected = payload_bits()
    detector = DctDetector(expected)
    return detector


# Lazy factory for the registered-but-unimplemented RivaGAN reader.
def _make_rivagan():
    """Build the RivaGAN stub (raises NotImplementedError on detect)."""
    from .rivagan_detector import RivaGanDetector

    detector = RivaGanDetector()
    return detector


# Register both names now; bodies run only when get_detector() is called.
register_detector("dwtDct", _make_dwt)
register_detector("rivaGan", _make_rivagan)


# Runs the named detector on the image at path.
# Never raises: missing library, missing file, or decode errors all become
# an "absent" finding with a clear reason (a broken checker must never
# crash the 3-checker pipeline or fake a positive).
def detect(path, detector_name: str = "dwtDct") -> WatermarkResult:
    """Detect our watermark in path with the named detector."""
    # Step 1: resolve the detector (unknown names fail loudly: our bug).
    # A missing heavy library is NOT our bug (pre-install runs), so it
    # becomes an honest "absent" with a clear reason instead of a crash.
    try:
        detector = get_detector(detector_name)
    except ImportError as exc:
        return WatermarkResult(
            found=False,
            ber=1.0,
            reason=f"watermark library not installed: {exc}",
            detector=detector_name,
        )

    # Step 2: run it, converting any failure into an honest "absent".
    try:
        result = detector.detect(path)
    except NotImplementedError:
        raise
    except Exception as exc:
        return WatermarkResult(
            found=False,
            ber=1.0,
            reason=f"detector failed safely: {exc}",
            detector=detector_name,
        )
    return result


# Converts a finding into the 3-number opinion the mixer understands.
def to_mass(result: WatermarkResult) -> MassFunction:
    """Map a WatermarkResult to a MassFunction.

    Found stamps start at (0, 0.90, 0.10) and are discounted by their own
    bit error rate: damage automatically demotes confidence (BER 0.2 still
    verifies; BER 0.3 slides to Suspicious via classify()). Absence leans
    weakly real. Unknown statuses fail loudly (programmer bug).
    """
    # Guard: a BER outside [0,1] is a programmer bug, fail loudly.
    if result.ber < 0.0 or result.ber > 1.0:
        raise ValueError(f"BER out of range [0,1]: {result.ber}")

    # Case 1: stamp found -> strong AI evidence, softened by damage.
    if result.found:
        base = MassFunction(
            m_auth=0.0,
            m_synth=VERIFIED_SYNTH,
            m_uncert=VERIFIED_UNCERT,
        )
        softened = discount(base, result.ber)
        return softened

    # Case 2: no stamp -> weak real lean, mostly unsure.
    # "Not stamped by us" is NOT evidence of authenticity.
    return MassFunction(
        m_auth=ABSENT_AUTH,
        m_synth=0.0,
        m_uncert=ABSENT_UNCERT,
    )
