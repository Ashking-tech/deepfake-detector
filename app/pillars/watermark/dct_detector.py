"""dwtDct detector: frequency-domain watermark reader (primary, Step 3).

Plain story: the magic bits are hidden in the image's frequency content
(DWT + DCT coefficients), not in visible pixels. This reader reverses that:
it recomputes the frequencies, reads 64 bits back out, and counts how many
differ from our magic. Small differences = our code, survived some damage.
~50% differences = random noise, nothing embedded.

Library: `invisible-watermark` (pip, open). dwtDct needs NO deep weights
and runs on CPU in ~100ms. NOTE: importing this module imports imwatermark,
which imports torch (its rivaGan submodule needs it) — so torch must be
INSTALLED to use this detector, even though dwtDct never runs it on GPU.
Prefer CPU-only torch to keep the install small (see README quickstart).

Known weakness (measured in the library's own docs): dwtDct FAILS on crop
and resize, passes JPEG / noise / brightness. Under crop, this detector
honestly reports absent (95% unsure) and forensics must carry the verdict.
RivaGAN (rivagan_detector.py) fixes crop; Stable Signature is parked as
future work because it needs the generator's key, which we don't have.
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
