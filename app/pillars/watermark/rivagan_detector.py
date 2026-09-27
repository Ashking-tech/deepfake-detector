"""RivaGAN detector stub: crop-robust upgrade path (Phase 3.5, NOT Step 3).

Why this file exists as a stub: the library's own measurements show
dwtDct FAILS on crop attacks while RivaGAN PASSES them (crop 7x5 test).
RivaGAN ships inside the same `invisible-watermark` pip package
(onnx weights bundled, needs onnxruntime + torch), so enabling it later
means implementing ONE class below and registering it — detect() and
to_mass() in detect.py never change.

Differences from dwtDct the implementer must respect:
  1. RivaGAN reads exactly 32 bits, not 64 (the library raises otherwise).
     Use a separate 32-bit magic: the first 4 ASCII bytes of our magic,
     i.e. "DEPA" (32 bits), documented in detect.py as RIVAGAN_MAGIC.
  2. Before first use, call WatermarkDecoder.loadModel() once (loads the
     bundled onnx encoder/decoder into memory).
  3. Encoder side: WatermarkEncoder().set_watermark('bits', bits32), then
     .encode(bgr, 'rivaGan'). Slower than dwtDct (~1s vs ~70ms at 600px),
     so keep it behind the registry, not as default.

Stable Signature (Meta) is deliberately NOT this file: it answers "did
THIS SPECIFIC generator make this image?" and needs that generator's key,
which we don't have. See README §6.
"""

from .base import WatermarkDetector, WatermarkResult


# Placeholder for the future RivaGAN reader. Registered in detect.py so
# get_detector("rivaGan") resolves, but calling detect() fails loudly with
# instructions instead of silently pretending to work.
class RivaGanDetector(WatermarkDetector):
    """32-bit neural watermark reader (not implemented yet)."""

    name = "rivaGan"
    payload_bits = 32

    def detect(self, path) -> WatermarkResult:
        """Always raises: implement per the module docstring first."""
        raise NotImplementedError(
            "RivaGAN detector not implemented yet (Phase 3.5). "
            "See rivagan_detector.py module docstring for the recipe."
        )
