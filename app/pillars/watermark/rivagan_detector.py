"""RivaGAN detector: neural watermark reader (primary, Step 3).

Plain story: a small neural network (bundled inside the
`invisible-watermark` pip package as onnx weights) reads 32 hidden bits out
of any image. We compare those bits to our magic "DEPA". Close = our stamp
survived. Far = nothing stamped here.

Why RivaGAN and not dwtDct (the original plan): dwtDct was measured BROKEN
in our dependency stack (opencv 5 / numpy 2 era) — its decoder returns
constant all-ones on every input, watermarked or not, so it carries zero
information (see dct_detector.py for the diagnosis). RivaGAN was measured
PERFECT on the same machine: BER 0.0 lossless, 0.0 after JPEG q95, 0.0
after JPEG q60, 0.0 after 50% center-crop, ~0.41 on clean images (random,
as a working blind decoder should output). Bonus: it also fixes dwtDct's
crop weakness.

Needs: invisible-watermark + onnxruntime + torch installed, any CPU.
The onnx models ship inside the pip package (loadModel finds them).
"""

import cv2

from imwatermark import WatermarkDecoder

from .base import WatermarkDetector, WatermarkResult

# Largest bit error rate that still counts as "found".
# Reasoning: random garbage decodes to BER ~0.5 (std ~0.09 over 32 bits),
# so BER <= 0.15 is essentially impossible by chance (~1 in 50k), while
# real damage measured so far (JPEG q60, 50% crop) costs ZERO bits.
MAX_BER_FOR_FOUND = 0.15

# RivaGAN reads exactly 32 bits per image (the library raises otherwise).
# This must match the magic our fixture tool embeds (RIVAGAN magic).
PAYLOAD_BITS = 32

# Tracks whether the bundled onnx models are loaded (once per process).
_models_loaded = False


# Neural reader for our 32-bit magic, via invisible-watermark's RivaGAN.
class RivaGanDetector(WatermarkDetector):
    """Reads our magic with the RivaGAN onnx decoder."""

    name = "rivaGan"
    payload_bits = PAYLOAD_BITS

    # Builds a decoder for 32-bit payloads and loads bundled onnx models.
    def __init__(self, expected_bits):
        """Store our magic bits and load models (first build pays ~seconds).

        expected_bits: list of 32 ints (0/1), e.g. RIVAGAN "DEPA" bits.
        """
        if len(expected_bits) != PAYLOAD_BITS:
            raise ValueError(
                f"expected {PAYLOAD_BITS} magic bits, got {len(expected_bits)}"
            )
        expected_copy = list(expected_bits)
        self.expected = expected_copy
        self._load_models_once()
        self.decoder = WatermarkDecoder("bits", PAYLOAD_BITS)

    # Loads the bundled onnx encoder/decoder exactly once per process.
    @classmethod
    def _load_models_once(cls):
        """Call WatermarkDecoder.loadModel() on first use only."""
        global _models_loaded
        if _models_loaded:
            return
        WatermarkDecoder.loadModel()
        _models_loaded = True

    # Decodes the image and counts mismatches against our magic.
    def detect(self, path) -> WatermarkResult:
        """Decode path with RivaGAN, return found + BER. Never raises."""
        # Step 1: load pixels. The library wants a BGR numpy array.
        image = cv2.imread(str(path))
        if image is None:
            return WatermarkResult(
                found=False,
                ber=1.0,
                reason=f"could not read image: {path}",
                detector=self.name,
            )

        # Step 2: decode 32 raw bits out of the image.
        try:
            raw_bits = self.decoder.decode(image, "rivaGan")
        except Exception as exc:
            return WatermarkResult(
                found=False,
                ber=1.0,
                reason=f"rivaGan decode failed: {exc}",
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
                reason=f"magic matched with {mismatches}/32 bit errors",
                detector=self.name,
            )

        # Step 5: ~40-50% errors means random noise, nothing embedded.
        return WatermarkResult(
            found=False,
            ber=ber,
            reason=f"BER {ber:.2f} looks like random noise, no watermark",
            detector=self.name,
        )
