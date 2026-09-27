# Fixture sources (Phase 0, image-only MVP)

- `real_01.jpg` — real photo via https://picsum.photos/seed/deep-real1/1024/768
  (Picsum serves Unsplash-sourced images under the Unsplash license; freely usable).
- `real_02.png` — real photo via https://picsum.photos/seed/deep-real2/1024/768,
  converted to PNG. Same license as above.
- `synthetic_01.png` — PROCEDURAL PLACEHOLDER, generated locally with
  numpy+Pillow (gradient + checker + noise, seed 7). NOT a real generator
  output. Replace with a real Stable Diffusion sample (own generation, so you
  own the rights) before any eval claims in Phase 7.
- `stripped_01.jpg` — derived from `real_01.jpg` via Pillow re-save (drops
  EXIF; simulates Instagram/CDN metadata stripping).
- `recompressed_01.jpg` — derived from `real_01.jpg`, JPEG quality=60
  (simulates social re-encode).

Pending for Phase 2 (NOT faked here):
- `c2pa_valid_camera.jpg` / `c2pa_valid_ai.png` — DONE in Step 2, self-signed
  with `tools/make_c2pa_fixtures.py` (throwaway self-signed ES256 cert,
  generated fresh per run in a temp dir, key destroyed afterwards).
  Camera fixture declares `c2pa.created` + `.../c2pa/captured`;
  AI fixture declares `c2pa.created` + `.../c2pa/trainedAlgorithmicMedia`.
  Both read back as validation_state "Valid" with the expected
  (non-trusted, self-signed) `signingCredential.untrusted` code, which is
  why the trust check stays OFF until Phase 8. Never commit fake C2PA bytes
  (these are real signatures, just untrusted test keys).

## Step 3 watermark fixtures (RivaGAN, `tools/make_watermark_fixtures.py`)

- `wm_ai_01.jpg` — from `real_01.jpg`, 32-bit magic "DEPA" embedded,
  saved JPEG q=95. Decodes BER 0.0.
- `wm_ai_01_jpg60.jpg` — the stamped pixels re-saved at JPEG q=60.
  Decodes BER 0.0 (RivaGAN is JPEG-proof at this level).
- `wm_ai_02.png` — from `real_02.png` (NOT synthetic_01: measured 2026-09,
  our procedural checkerboard defeats the RivaGAN decoder itself, BER 0.47
  even in-memory — extreme local contrast breaks it; natural photos decode
  at 0.0). The synthetic placeholder stays unstamped as a no-stamp control
  until Phase 7 replaces it with a real SD sample.
- `wm_ai_01_rot30.jpg` — stamped pixels rotated 30°. Decodes BER ~0.28
  (above the 0.15 gate) → honest absent. Kill-demo: rotation destroys
  neural watermarks, and the checker must say "unsure", never positive.

Retired: dwtDct was the original plan but measured broken in our stack
(opencv 5 / numpy 2 era, 2021 library): constant all-ones output on every
input, zero information. Kept in-tree with the diagnosis, never default.
