"""Image preparation for Checker 3 (picture detective).

Plain story: every model below wants the same thing — a square RGB picture
at 224x224 pixels. This module does only that boring-but-critical shaping
with PIL + numpy (no deep libraries here, so it always imports). Each
model then applies its OWN normalization in its own file, because
ConvNeXt and DINO were trained with different recipes.
"""

import numpy as np
from PIL import Image

# Every backbone in Step 4 eats 224x224 squares. One size for all means
# the FFT branch and both neural nets all look at the same pixels.
TARGET_SIZE = 224


# Loads an image file as RGB pixels. Returns None (never raises) when the
# file is missing or unreadable — a broken input must not crash the pillar.
def load_image(path):
    """Open path as RGB. Returns PIL image, or None on failure."""
    # Step 1: open the file.
    try:
        image = Image.open(path)
    except Exception:
        return None

    # Step 2: force RGB (drops alpha channels, expands grayscale).
    try:
        rgb = image.convert("RGB")
    except Exception:
        return None

    return rgb


# Cuts the center square out of a picture (keeps the middle, drops edges).
def square_crop(image):
    """Center-crop a PIL image to a square. Returns a new PIL image."""
    # Step 1: find the shorter side (that sets the square size).
    width, height = image.size
    if width < height:
        side = width
    else:
        side = height

    # Step 2: center the square.
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    # Step 3: cut it out.
    square = image.crop((left, top, right, bottom))
    return square


# Full pipeline: load -> RGB -> center square -> 224x224. One call that
# every backbone shares, so nobody accidentally trains on different pixels.
def prepare(image):
    """Crop + resize a PIL RGB image to TARGET_SIZE square."""
    # Step 1: center square (keeps aspect honest, no squishing).
    square = square_crop(image)

    # Step 2: resize to what the models expect.
    small = square.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    return small


# Turns a prepared PIL image into a float array in [0,1], channels last.
def to_array(image):
    """PIL RGB -> numpy float64 array shaped (H, W, 3) in [0,1]."""
    # Step 1: raw bytes to numbers.
    raw = np.asarray(image, dtype=np.float64)

    # Step 2: scale 0..255 down to 0..1.
    scaled = raw / 255.0
    return scaled
