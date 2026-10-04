# Step 6 attack-lite report

Generated: 2026-10-04 16:43 UTC

Environment (pillars missing a library vote unsure):

- c2pa (Checker 1): MISSING
- watermark stack (Checker 2): MISSING
- torch backbones (Checker 3 deep): MISSING (FFT heuristic only)

Summary: verdict kept after attack 55/55, honest Unknown 45/55.

| Image | Attack | Verdict (driver) |
|---|---|---|
| c2pa_valid_ai.png | original | Suspicious (forensics) |
| c2pa_valid_ai.png | strip | Suspicious (forensics) |
| c2pa_valid_ai.png | jpg70 | Suspicious (forensics) |
| c2pa_valid_ai.png | jpg50 | Suspicious (forensics) |
| c2pa_valid_ai.png | crop10 | Suspicious (forensics) |
| c2pa_valid_ai.png | resize50 | Suspicious (forensics) |
| c2pa_valid_camera.jpg | original | Unknown (forensics) |
| c2pa_valid_camera.jpg | strip | Unknown (forensics) |
| c2pa_valid_camera.jpg | jpg70 | Unknown (forensics) |
| c2pa_valid_camera.jpg | jpg50 | Unknown (forensics) |
| c2pa_valid_camera.jpg | crop10 | Unknown (forensics) |
| c2pa_valid_camera.jpg | resize50 | Unknown (forensics) |
| real_01.jpg | original | Unknown (forensics) |
| real_01.jpg | strip | Unknown (forensics) |
| real_01.jpg | jpg70 | Unknown (forensics) |
| real_01.jpg | jpg50 | Unknown (forensics) |
| real_01.jpg | crop10 | Unknown (forensics) |
| real_01.jpg | resize50 | Unknown (forensics) |
| real_02.png | original | Unknown (forensics) |
| real_02.png | strip | Unknown (forensics) |
| real_02.png | jpg70 | Unknown (forensics) |
| real_02.png | jpg50 | Unknown (forensics) |
| real_02.png | crop10 | Unknown (forensics) |
| real_02.png | resize50 | Unknown (forensics) |
| recompressed_01.jpg | original | Unknown (forensics) |
| recompressed_01.jpg | strip | Unknown (forensics) |
| recompressed_01.jpg | jpg70 | Unknown (forensics) |
| recompressed_01.jpg | jpg50 | Unknown (forensics) |
| recompressed_01.jpg | crop10 | Unknown (forensics) |
| recompressed_01.jpg | resize50 | Unknown (forensics) |
| stripped_01.jpg | original | Unknown (forensics) |
| stripped_01.jpg | strip | Unknown (forensics) |
| stripped_01.jpg | jpg70 | Unknown (forensics) |
| stripped_01.jpg | jpg50 | Unknown (forensics) |
| stripped_01.jpg | crop10 | Unknown (forensics) |
| stripped_01.jpg | resize50 | Unknown (forensics) |
| synthetic_01.png | original | Suspicious (forensics) |
| synthetic_01.png | strip | Suspicious (forensics) |
| synthetic_01.png | jpg70 | Suspicious (forensics) |
| synthetic_01.png | jpg50 | Suspicious (forensics) |
| synthetic_01.png | crop10 | Suspicious (forensics) |
| synthetic_01.png | resize50 | Suspicious (forensics) |
| wm_ai_01.jpg | original | Unknown (forensics) |
| wm_ai_01.jpg | strip | Unknown (forensics) |
| wm_ai_01.jpg | jpg70 | Unknown (forensics) |
| wm_ai_01.jpg | jpg50 | Unknown (forensics) |
| wm_ai_01.jpg | crop10 | Unknown (forensics) |
| wm_ai_01.jpg | resize50 | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | original | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | strip | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | jpg70 | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | jpg50 | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | crop10 | Unknown (forensics) |
| wm_ai_01_jpg60.jpg | resize50 | Unknown (forensics) |
| wm_ai_01_rot30.jpg | original | Unknown (forensics) |
| wm_ai_01_rot30.jpg | strip | Unknown (forensics) |
| wm_ai_01_rot30.jpg | jpg70 | Unknown (forensics) |
| wm_ai_01_rot30.jpg | jpg50 | Unknown (forensics) |
| wm_ai_01_rot30.jpg | crop10 | Unknown (forensics) |
| wm_ai_01_rot30.jpg | resize50 | Unknown (forensics) |
| wm_ai_02.png | original | Unknown (forensics) |
| wm_ai_02.png | strip | Unknown (forensics) |
| wm_ai_02.png | jpg70 | Unknown (forensics) |
| wm_ai_02.png | jpg50 | Unknown (forensics) |
| wm_ai_02.png | crop10 | Unknown (forensics) |
| wm_ai_02.png | resize50 | Unknown (forensics) |

Reading guide: same verdict as `original` = pillar survived.
`Unknown` after attack = honest abstain (attack worked, we admit it).
A flip (e.g. Suspicious -> Unknown) is graceful damage, not a lie.
