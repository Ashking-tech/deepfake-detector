"""One-shot dev tool: stamps the watermark test fixtures.

What it makes (in tests/fixtures/, all derived from UNSIGNED sources so
pillar 2 is tested in isolation from C2PA):
  wm_ai_01.jpg       from real_01.jpg, magic embedded, saved JPEG q=95
  wm_ai_01_jpg60.jpg the stamped image re-saved at JPEG q=60 (damage demo)
  wm_ai_02.png       from synthetic_01.png, magic embedded, lossless PNG

Each output is immediately decoded back; the script refuses to finish
unless the exact fixture decodes nearly perfectly (BER <= 0.05).

Needs: invisible-watermark + opencv installed (post-install step).
Usage (inside .venv):
  python tools/make_watermark_fixtures.py
"""

import pathlib

import cv2

from app.pillars.watermark.detect import PAYLOAD_BITS, payload_bits

# Where the input images live and where stamped copies go.
FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# JPEG quality for the "exact" fixture (high: near-lossless round trip).
EXACT_QUALITY = 95

# JPEG quality for the "damaged" fixture (social-media-like recompression).
DAMAGED_QUALITY = 60

# The exact fixture must decode almost perfectly, or embedding is broken.
MAX_EXACT_BER = 0.05


# Embeds our magic into one image with dwtDct and saves the result.
def stamp_image(src_name, dst_name, quality=None):
    """Embed PAYLOAD magic into src_name, save as dst_name.

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

    # Step 3: configure the encoder with our 64-bit magic.
    bits = payload_bits()
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", bits)

    # Step 4: embed and save.
    stamped = encoder.encode(image, "dwtDct")
    dst_path = FIXTURES / dst_name
    if quality is None:
        saved = cv2.imwrite(str(dst_path), stamped)
    else:
        saved = cv2.imwrite(str(dst_path), stamped, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not saved:
        raise SystemExit(f"cannot write stamped image: {dst_path}")
    print(dst_name, "stamped")
    return stamped


# Decodes one stamped file and reports its BER against our magic.
def check_back(dst_name):
    """Decode dst_name, print BER, return it."""
    # Step 1: late import, same reason as above.
    from imwatermark import WatermarkDecoder

    # Step 2: decode 64 raw bits.
    image = cv2.imread(str(FIXTURES / dst_name))
    decoder = WatermarkDecoder("bits", PAYLOAD_BITS)
    raw = decoder.decode(image, "dwtDct")
    decoded = [int(bit) % 2 for bit in raw]

    # Step 3: count mismatches against our magic.
    expected = payload_bits()
    mismatches = 0
    for got, want in zip(decoded, expected):
        if got != want:
            mismatches = mismatches + 1
    ber = mismatches / PAYLOAD_BITS
    print(dst_name, f"-> BER {ber:.3f} ({mismatches}/64 errors)")
    return ber


# Stamps all fixtures and verifies the exact one decodes cleanly.
def main():
    """Embed magic into 3 fixtures, damage one, verify, print proof."""
    # Step 1: exact JPEG fixture (near-lossless round trip).
    stamp_image("real_01.jpg", "wm_ai_01.jpg", quality=EXACT_QUALITY)

    # Step 2: damaged copy (re-encode the STAMPED pixels at q=60).
    stamped_path = FIXTURES / "wm_ai_01.jpg"
    damaged = cv2.imread(str(stamped_path))
    cv2.imwrite(
        str(FIXTURES / "wm_ai_01_jpg60.jpg"),
        damaged,
        [cv2.IMWRITE_JPEG_QUALITY, DAMAGED_QUALITY],
    )
    print("wm_ai_01_jpg60.jpg re-encoded at q=60")

    # Step 3: lossless PNG fixture from the synthetic placeholder.
    stamp_image("synthetic_01.png", "wm_ai_02.png", quality=None)

    # Step 4: verify the exact fixture decodes nearly perfectly.
    ber = check_back("wm_ai_01.jpg")
    if ber > MAX_EXACT_BER:
        raise SystemExit(f"exact fixture BER {ber:.3f} too high, embedding broken")

    # Step 5: report (don't gate on) the damaged fixture's BER.
    check_back("wm_ai_01_jpg60.jpg")
    check_back("wm_ai_02.png")
    print("done: watermark fixtures in tests/fixtures/")


if __name__ == "__main__":
    main()
