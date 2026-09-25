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
