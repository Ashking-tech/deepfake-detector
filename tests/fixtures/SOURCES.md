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
- `c2pa_valid_camera.jpg` / `c2pa_valid_ai.jpg` — will come from C2PA spec
  test vectors or self-signed via `c2patool`. Never commit fake C2PA bytes.
