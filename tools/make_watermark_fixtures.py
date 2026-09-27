"""One-shot dev tool: stamps the watermark test fixtures.

What it makes (in tests/fixtures/, all derived from UNSIGNED sources so
pillar 2 is tested in isolation from C2PA):
  wm_ai_01.jpg       from real_01.jpg, RivaGAN magic embedded, JPEG q=95
  wm_ai_01_jpg60.jpg the stamped image re-saved at JPEG q=60 (damage demo)
  wm_ai_02.png       from real_02.png, magic embedded, lossless PNG
  wm_ai_01_rot30.jpg the stamped image rotated 30 degrees (kill demo:
                     rotation destroys neural watermarks -> honest absent)

Why wm_ai_02 comes from real_02, not synthetic_01: measured 2026-09 —
our procedural checkerboard defeats the RivaGAN decoder itself (BER 0.47
even in-memory; extreme local contrast breaks it), while natural photos
decode at BER 0.0. The synthetic placeholder stays unstamped as a
no-stamp control until Phase 7 replaces it with a real SD sample.

Each output is decoded back and its BER printed. The script refuses to
finish unless the exact fixture decodes nearly perfectly (BER <= 0.05);
the damaged copies only report (different attacks, different fates).

Needs: invisible-watermark + onnxruntime + opencv installed.
Usage (inside .venv):
  python tools/make_watermark_fixtures.py
"""

import pathlib
import sys

# Make the project root importable no matter how this script is launched:
# running `python tools/x.py` puts tools/ (not the root) on sys.path.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from app.pillars.watermark.detect import RIVAGAN_BITS, rivagan_bits

# Where the input images live and where stamped copies go.
FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# JPEG quality for the "exact" fixture (high: near-lossless round trip).
EXACT_QUALITY = 95

# JPEG quality for the "damaged" fixture (social-media-like recompression).
DAMAGED_QUALITY = 60

# Rotation angle for the "kill" fixture (destroys the watermark on purpose).
KILL_ANGLE = 30

# The exact fixture must decode almost perfectly, or embedding is broken.
MAX_EXACT_BER = 0.05


# Embeds our magic into one image with RivaGAN and saves the result.
def stamp_image(src_name, dst_name, quality=None):
    """Embed RIVAGAN magic into src_name, save as dst_name.

    quality=None saves losslessly (PNG path); otherwise JPEG quality.
    Returns the stamped BGR array.
    """
    # Step 1: late import, so this tool (not the pillar) owns the dependency.
    from imwatermark import WatermarkEncoder

    # Step 2: load source pixels (BGR, as the library expects).
    src_path = FIXTURES / src_name
    image = cv2.imread(str(src_path))
    if image is None:
        raise SystemExit(f"cannot read source image: {src_path}")

    # Step 3: configure the encoder with our 32-bit magic.
    bits = rivagan_bits()
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", bits)

    # Step 4: embed and save.
    stamped = encoder.encode(image, "rivaGan")
    dst_path = FIXTURES / dst_name
    if quality is None:
        saved = cv2.imwrite(str(dst_path), stamped)
    else:
        saved = cv2.imwrite(str(dst_path), stamped, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not saved:
        raise SystemExit(f"cannot write stamped image: {dst_path}")
    print(dst_name, "stamped")
    return stamped


# Decodes one file with RivaGAN and reports its BER against our magic.
def check_back(dst_name):
    """Decode dst_name, print BER, return it."""
    # Step 1: late import, same reason as above.
    from imwatermark import WatermarkDecoder

    # Step 2: decode 32 raw bits.
    image = cv2.imread(str(FIXTURES / dst_name))
    decoder = WatermarkDecoder("bits", RIVAGAN_BITS)
    raw = decoder.decode(image, "rivaGan")
    decoded = [int(bit) % 2 for bit in raw]

    # Step 3: count mismatches against our magic.
    expected = rivagan_bits()
    mismatches = 0
    for got, want in zip(decoded, expected):
        if got != want:
            mismatches = mismatches + 1
    ber = mismatches / RIVAGAN_BITS
    print(dst_name, f"-> BER {ber:.3f} ({mismatches}/32 errors)")
    return ber


# Rotates an image by KILL_ANGLE degrees (destroys neural watermarks).
def rotate_image(src_name, dst_name):
    """Rotate src_name, save as dst_name (attack simulation)."""
    # Step 1: load the stamped pixels.
    image = cv2.imread(str(FIXTURES / src_name))

    # Step 2: build a rotation around the image center.
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, KILL_ANGLE, 1.0)

    # Step 3: warp and save (corners get black borders: extra damage).
    rotated = cv2.warpAffine(image, matrix, (width, height))
    cv2.imwrite(str(FIXTURES / dst_name), rotated)
    print(dst_name, f"rotated {KILL_ANGLE} degrees")


# Stamps all fixtures, damages two copies, verifies the exact one.
def main():
    """Embed magic, damage copies, verify, print proof."""
    # Step 0: load the bundled onnx models once (encoder needs them too).
    from imwatermark import WatermarkEncoder as _Enc

    _Enc.loadModel()
    print("rivaGan models loaded")

    # Step 1: exact JPEG fixture (near-lossless round trip).
    stamp_image("real_01.jpg", "wm_ai_01.jpg", quality=EXACT_QUALITY)

    # Step 2: damaged copy (re-encode the STAMPED pixels at q=60).
    stamped = cv2.imread(str(FIXTURES / "wm_ai_01.jpg"))
    cv2.imwrite(
        str(FIXTURES / "wm_ai_01_jpg60.jpg"),
        stamped,
        [cv2.IMWRITE_JPEG_QUALITY, DAMAGED_QUALITY],
    )
    print("wm_ai_01_jpg60.jpg re-encoded at q=60")

    # Step 3: lossless PNG fixture from the second real photo.
    stamp_image("real_02.png", "wm_ai_02.png", quality=None)

    # Step 4: kill copy (rotation destroys the stamp on purpose).
    rotate_image("wm_ai_01.jpg", "wm_ai_01_rot30.jpg")

    # Step 5: verify the exact fixture decodes nearly perfectly.
    ber = check_back("wm_ai_01.jpg")
    if ber > MAX_EXACT_BER:
        raise SystemExit(f"exact fixture BER {ber:.3f} too high, embedding broken")

    # Step 6: report (don't gate on) the damaged copies' BERs.
    check_back("wm_ai_01_jpg60.jpg")
    check_back("wm_ai_02.png")
    check_back("wm_ai_01_rot30.jpg")
    print("done: watermark fixtures in tests/fixtures/")


if __name__ == "__main__":
    main()
