"""dwtDct detector: frequency-domain watermark reader (RETIRED, kept on record).

STATUS 2026-09: MEASURED BROKEN in our dependency stack (opencv 5,
numpy 2, PyWavelets 1.8 era; library is 2021 abandonware). Diagnosis:
the decoder returns constant all-ones on EVERY input — watermarked,
clean, even in-memory roundtrips (32/64 errors on alternating bits).
Root cause: residue statistics of DWT approximation coefficients skew
past the 0.5*scale decision line under the new cv2 YUV scaling, so the
majority vote saturates at 1 and the channel carries zero information.
RivaGAN (rivagan_detector.py) measured PERFECT on the same machine and
is the default. This file stays registered so the failure is documented
in code, not forgotten — do NOT make it default without re-measuring.
"""

import cv2

from imwatermark import WatermarkDecoder

from .base import WatermarkDetector, WatermarkResult

# Largest bit error rate that still counts as "found".
# Reasoning: random garbage decodes to BER ~0.5 (std ~0.06 over 64 bits),
# so BER <= 0.20 is essentially impossible by chance, while real damage
# (JPEG, noise) typically costs only a few bits.
MAX_BER_FOR_FOUND = 0.20

# dwtDct reads exactly 64 bits per image. This must match the payload our
# fixture tool embeds (PAYLOAD_BITS in detect.py).
PAYLOAD_BITS = 64


# Frequency-domain reader for our 64-bit magic, via invisible-watermark.
class DctDetector(WatermarkDetector):
    """Reads our magic with the dwtDct algorithm."""

    name = "dwtDct"
    payload_bits = PAYLOAD_BITS

    # Builds a decoder for 64 'bits'-type payloads.
    def __init__(self, expected_bits):
        """Store our magic bits for later comparison.

        expected_bits: list of 64 ints (0/1), e.g. from payload_bits().
        """
        if len(expected_bits) != PAYLOAD_BITS:
            raise ValueError(
                f"expected {PAYLOAD_BITS} magic bits, got {len(expected_bits)}"
            )
        expected_copy = list(expected_bits)
        self.expected = expected_copy
        self.decoder = WatermarkDecoder("bits", PAYLOAD_BITS)

    # Decodes the image and counts mismatches against our magic.
    def detect(self, path) -> WatermarkResult:
        """Decode path with dwtDct, return found + BER. Never raises."""
        # Step 1: load pixels. The library wants a BGR numpy array.
        image = cv2.imread(str(path))
        if image is None:
            return WatermarkResult(
                found=False,
                ber=1.0,
                reason=f"could not read image: {path}",
                detector=self.name,
            )

        # Step 2: decode 64 raw bits out of the frequencies.
        try:
            raw_bits = self.decoder.decode(image, "dwtDct")
        except Exception as exc:
            return WatermarkResult(
                found=False,
                ber=1.0,
                reason=f"dwtDct decode failed: {exc}",
                detector=self.name,
            )

        # Step 3: count mismatches against our magic, one bit at a time.
        decoded = [int(bit) % 2 for bit in raw_bits]
        mismatches = 0
        for got, want in zip(decoded, self.expected):
            if got != want:
                mismatches = mismatches + 1
        ber = mismatches / PAYLOAD_BITS

        # Step 4: close enough counts as found (allows for damage).
        if ber <= MAX_BER_FOR_FOUND:
            return WatermarkResult(
                found=True,
                ber=ber,
                reason=f"magic matched with {mismatches}/64 bit errors",
                detector=self.name,
            )

        # Step 5: ~50% errors means random noise, nothing embedded.
        return WatermarkResult(
            found=False,
            ber=ber,
            reason=f"BER {ber:.2f} looks like random noise, no watermark",
            detector=self.name,
        )
