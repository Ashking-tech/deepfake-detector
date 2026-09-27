"""Checker 2: hidden code (watermark) protocol + verdict mapping.

Plain story: WE stamp our own images with a secret pattern. Later, the
checker decodes any image and measures how close the decoded bits are to
that pattern. Close = our stamp survived. Far = nothing stamped here.
This mirrors how SynthID works (shared secret key), except our key is our
own published test magic.

Default detector is RivaGAN (32-bit magic "DEPA"): measured perfect in our
stack (BER 0.0 through JPEG q60 and 50% crop). dwtDct stays registered for
the record but is BROKEN in our dependency era (see dct_detector.py) —
kept, not deleted, so the failure stays documented, not forgotten.

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

# Default detector name (RivaGAN: measured perfect, fixes crop weakness).
DEFAULT_DETECTOR = "rivaGan"

# Our magic for RivaGAN, as 4 ASCII characters = exactly 32 bits (its only
# supported length). First 4 bytes of the old 64-bit magic, so the family
# stays recognizable. Public on purpose: the statistical test
# (1/2^32 chance of random match at BER 0) is the security, not secrecy.
RIVAGAN_ASCII = "DEPA"
RIVAGAN_BITS = 32

# Legacy 64-bit magic for dwtDct ("DEPAUTH1"). Kept for the record;
# dwtDct is broken in our stack, so nothing uses this by default.
PAYLOAD_ASCII = "DEPAUTH1"
PAYLOAD_BITS = 64

# Mass shares for a perfectly decoded stamp (matches README's "verified").
VERIFIED_SYNTH = 0.90
VERIFIED_UNCERT = 0.10

# Mass shares when no stamp is found: weak real lean, mostly unsure.
ABSENT_AUTH = 0.05
ABSENT_UNCERT = 0.95


# Turns ASCII text into bits, most-significant bit first per byte.
def text_to_bits(text: str) -> list:
    """Encode ASCII text as a list of ints (0/1), 8 bits per char."""
    bits = []
    for char in text:
        code = ord(char)
        binary = format(code, "08b")
        for digit in binary:
            bits.append(int(digit))
    return bits


# Our 64-bit legacy magic as bits (dwtDct record only).
def payload_bits() -> list:
    """Encode PAYLOAD_ASCII as a list of 64 ints (0/1)."""
    return text_to_bits(PAYLOAD_ASCII)


# Our 32-bit RivaGAN magic as bits (the live protocol).
def rivagan_bits() -> list:
    """Encode RIVAGAN_ASCII as a list of 32 ints (0/1)."""
    return text_to_bits(RIVAGAN_ASCII)


# Lazy factory: imports the heavy library only when actually built, so
# `import detect` stays cheap and tests can guard on availability.
def _make_rivagan():
    """Build the default RivaGAN detector (imports imwatermark now)."""
    from .rivagan_detector import RivaGanDetector

    expected = rivagan_bits()
    detector = RivaGanDetector(expected)
    return detector


# Lazy factory for the legacy dwtDct reader (broken env, kept on record).
def _make_dwt():
    """Build the legacy dwtDct detector (imports imwatermark now)."""
    from .dct_detector import DctDetector

    expected = payload_bits()
    detector = DctDetector(expected)
    return detector


# Register both names now; bodies run only when get_detector() is called.
register_detector("rivaGan", _make_rivagan)
register_detector("dwtDct", _make_dwt)


# Runs the named detector on the image at path.
# Never raises: missing library, missing file, or decode errors all become
# an "absent" finding with a clear reason (a broken checker must never
# crash the 3-checker pipeline or fake a positive).
def detect(path, detector_name: str = DEFAULT_DETECTOR) -> WatermarkResult:
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
    weakly real. BER outside [0,1] fails loudly (programmer bug).
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
