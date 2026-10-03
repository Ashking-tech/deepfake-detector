"""DINOv2 safety net: frozen foundation backbone + untrained probe.

Plain story: DINOv2-Base learned "what natural images look like" from
142M images with no labels (86.6M params, ~346MB, Apache-2.0 from Meta
via HuggingFace). Research shows frozen DINO features + a tiny trained
classifier generalize to AI generators the classifier never saw — the
opposite of classic detectors that memorize one generator's quirks.
That generalization is exactly what our cross-generator gap needs.

STATUS: the backbone RUNS (verified CLS shapes), but the linear probe
is RANDOM (seeded for determinism) and flagged trained=False — so its
vote carries ZERO weight in detect.py until Phase 7 trains it on real
deepfake datasets. Same pattern as convnext.py: wire now, train later.

Checkpoint: facebook/dinov2-base (auto-downloaded to the HF cache on
first use, ~346MB safetensors).
"""

import numpy as np

# Length of the CLS understanding vector (ViT-Base embedding width).
EMBED_DIM = 768

# Random seed for the untrained probe (same garbage every run, so tests
# are deterministic and nobody mistakes it for a trained opinion).
PROBE_SEED = 0

# Cached backbone + processor (loaded once per process).
_backbone = None


# Loads the frozen DINOv2 backbone + its image processor once.
def _load_once():
    """Import transformers and build the frozen backbone. Cached globally."""
    # Step 1: late imports so `import dino_probe` stays cheap.
    from transformers import AutoImageProcessor, AutoModel

    # Step 2: use the module-global cache.
    global _backbone
    if _backbone is not None:
        return _backbone

    # Step 3: processor (resize/normalize recipe) + frozen backbone.
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.eval()

    _backbone = (processor, model)
    return _backbone


# Reads the CLS understanding vector for one prepared RGB image.
def cls_embed(pil_image):
    """Frozen DINOv2 CLS vector (768 numbers) for a PIL RGB image."""
    # Step 1: late torch import (heavy library, only when actually used).
    import torch

    # Step 2: load (or reuse) backbone + processor.
    processor, model = _load_once()

    # Step 3: processor handles resize + normalize (its own recipe).
    inputs = processor(images=pil_image, return_tensors="pt")

    # Step 4: frozen forward pass (no gradients, no learning).
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 5: first token = CLS = whole-image understanding.
    cls_vector = outputs.last_hidden_state[:, 0, :]
    vector = cls_vector[0].detach().cpu().numpy()
    return vector


# Random linear probe: deterministic garbage until Phase 7 trains it.
def probe_score(cls_vector):
    """Random-init probe vote. Returns (score, trained=False) always."""
    # Step 1: deterministic random weights (same every run).
    rng = np.random.default_rng(PROBE_SEED)
    weights = rng.normal(0.0, 0.01, size=(EMBED_DIM,))

    # Step 2: dot product + squash (a vote nobody should trust yet).
    logit = float(np.dot(weights, cls_vector))
    score = 1.0 / (1.0 + np.exp(-logit))

    # Step 3: the flag that keeps this vote at zero weight in detect.py.
    trained = False
    return float(score), trained
