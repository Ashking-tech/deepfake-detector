"""ConvNeXt-Tiny eyes: frozen feature extractor (wired, zero-weighted).

Plain story: ConvNeXt-Tiny is a modern image network pre-trained on
ImageNet (28.6M params, ~115MB, Apache-2.0 via timm). We chop off its
1000-class photo-label head and keep the 768-number "understanding"
vector it produces for any image. In Phase 7 that vector feeds a small
real/fake classifier trained on deepfake datasets.

STATUS: the extractor RUNS (verified shapes), but no real/fake head is
trained yet — so its output carries ZERO weight in detect.py (logged in
details for inspection, never voted on). Wiring it now means Phase 7
trains a head without touching the pipeline.

Checkpoint: timm/convnext_tiny.in12k_ft_in1k (auto-downloaded to the
timm/HF cache on first use, ~115MB).
"""

import numpy as np

# ImageNet recipe this checkpoint was trained with (from its own config).
# Must match exactly, or the features are garbage.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Length of the understanding vector ( ConvNeXt-Tiny stage-4 channels).
EMBED_DIM = 768

# Cached model (loaded once per process; weights download on first use).
_model = None


# Loads the headless ConvNeXt-Tiny once, in eval mode, on CPU.
def _load_once():
    """Import timm and build the feature extractor. Cached globally."""
    # Step 1: late import so `import convnext` stays cheap without timm.
    import timm
    import torch

    # Step 2: use the module-global cache.
    global _model
    if _model is not None:
        return _model

    # Step 3: headless = no classification head, outputs the 768 vector.
    model = timm.create_model(
        "convnext_tiny.in12k_ft_in1k",
        pretrained=True,
        num_classes=0,
    )

    # Step 4: eval mode (freezes dropout/batchnorm behavior).
    model = model.eval()

    _model = model
    return _model


# Turns a prepared 224x224 RGB array into the 768-number understanding.
def embed(rgb01):
    """Extract frozen ConvNeXt features. Input: float array in [0,1]."""
    # Step 1: late torch import (heavy library, only when actually used).
    import torch

    # Step 2: load (or reuse) the frozen model.
    model = _load_once()

    # Step 3: channels-first, then ImageNet normalize, one at a time.
    tensor = torch.from_numpy(np.asarray(rgb01, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    # Step 4: frozen forward pass (no gradients, no learning).
    with torch.no_grad():
        features = model(tensor)

    # Step 5: back to plain numpy.
    vector = features[0].detach().cpu().numpy()
    return vector
