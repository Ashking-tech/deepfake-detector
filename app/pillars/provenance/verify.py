"""Checker 1: history file (C2PA provenance).

Plain story: some images carry a hidden history file saying "made by camera
X" or "made by AI". This checker reads it with the official c2pa-python
library (we never hand-parse crypto) and reports one of 5 plain findings.
If the file is missing (Instagram and friends delete it), the answer is an
honest "I don't know" — never "real".

Trust-list note (read this before production): signature validity and signer
trust are TWO separate questions. This module answers question 1 (is the
file intact?) via validation_state. Question 2 (do we trust who signed it?)
needs official trust anchors loaded through c2pa Settings/Context and shows
up as the code "signingCredential.trusted" inside get_validation_results().
Our test fixtures are self-signed, so they correctly report
"signingCredential.untrusted" — therefore trust checking stays OFF until
Phase 8 wires real anchors in.
"""

import json
from dataclasses import dataclass

import c2pa

from app.fusion.mass import MassFunction

# The 5 possible findings. Keep them as plain words, not booleans, so logs
# stay readable.
NO_MANIFEST = "no_manifest"
VALID_AI = "valid_ai"
VALID_CAMERA = "valid_camera"
TAMPERED = "tampered"
UNREADABLE = "unreadable"

# Marker inside a digitalSourceType value meaning "an AI model made this".
# Compared in lowercase, so casing never matters.
AI_SOURCE_MARKER = "trainedalgorithmicmedia"

# Mass shares for a solid cryptographic proof (5% left unsure: humility).
STRONG_PROOF = 0.95
STRONG_PROOF_UNCERT = 0.05

# Mass shares for a broken/faked history file: a hint of meddling (50% AI)
# but only a hint, so half stays unsure. NOT proof.
TAMPER_SYNTH = 0.5
TAMPER_UNCERT = 0.5


@dataclass
class ProvenanceResult:
    """Plain finding from the history-file check.

    status: one of the 5 NO_MANIFEST / VALID_AI / VALID_CAMERA / TAMPERED /
        UNREADABLE words above.
    reason: one-line human explanation, goes straight into logs and UI.
    ai_claim: True when the history file itself declares AI involvement.
    validation_state: raw c2pa verdict ("Valid"/"Invalid"/..., None if none).
    """

    status: str
    reason: str
    ai_claim: bool = False
    validation_state: str | None = None


# Opens an image and asks c2pa about its history file.
# Never raises: any failure becomes an UNREADABLE finding, because one
# broken checker must never crash the whole 3-checker pipeline.
def check(path) -> ProvenanceResult:
    """Read the C2PA history file of the image at path.

    Returns a ProvenanceResult (never raises).
    """
    # Step 1: ask c2pa for a reader. None = no history file embedded.
    try:
        reader = c2pa.Reader.try_create(str(path))
    except Exception as exc:
        # c2pa itself choked (not "no manifest", something worse).
        return ProvenanceResult(
            status=UNREADABLE,
            reason=f"c2pa reader failed: {exc}",
        )

    # Step 2: no history file is the common social-media case.
    if reader is None:
        return ProvenanceResult(
            status=NO_MANIFEST,
            reason="no JUMBF manifest embedded (often stripped by platforms)",
        )

    # Step 3: a history file exists — read its verdict and its content.
    try:
        state = reader.get_validation_state()
        store_text = reader.json()
    except Exception as exc:
        return ProvenanceResult(
            status=UNREADABLE,
            reason=f"could not read manifest contents: {exc}",
        )
    finally:
        reader.close()

    # Step 4: look inside the manifest for an AI declaration.
    store = json.loads(store_text)
    source_types = find_digital_source_types(store)
    ai_claim = has_ai_claim(source_types)

    # Step 5: intact file + AI declared -> proof of AI origin.
    if state == "Valid" or state == "Trusted":
        if ai_claim:
            return ProvenanceResult(
                status=VALID_AI,
                reason="valid manifest declares AI generation",
                ai_claim=True,
                validation_state=state,
            )

        # Step 6: intact file, no AI declared -> proof of camera origin.
        return ProvenanceResult(
            status=VALID_CAMERA,
            reason="valid manifest, no AI declaration found",
            ai_claim=False,
            validation_state=state,
        )

    # Step 7: anything else (Invalid, None, ...) means the file or the
    # pixels were messed with after signing -> hint, not proof.
    return ProvenanceResult(
        status=TAMPERED,
        reason=f"manifest failed validation (state={state})",
        ai_claim=ai_claim,
        validation_state=state,
    )


# Converts a finding into the 3-number opinion the mixer understands.
def to_mass(result: ProvenanceResult) -> MassFunction:
    """Map a ProvenanceResult to a MassFunction. Raises ValueError on an
    unknown status (that is a programmer bug, fail loudly in tests)."""
    # Case 1: solid proof of AI origin.
    if result.status == VALID_AI:
        return MassFunction(
            m_auth=0.0,
            m_synth=STRONG_PROOF,
            m_uncert=STRONG_PROOF_UNCERT,
        )

    # Case 2: solid proof of camera origin.
    if result.status == VALID_CAMERA:
        return MassFunction(
            m_auth=STRONG_PROOF,
            m_synth=0.0,
            m_uncert=STRONG_PROOF_UNCERT,
        )

    # Case 3: broken/faked file -> weak lean toward AI, half unsure.
    if result.status == TAMPERED:
        return MassFunction(
            m_auth=0.0,
            m_synth=TAMPER_SYNTH,
            m_uncert=TAMPER_UNCERT,
        )

    # Case 4: no file, or unreadable -> total ignorance.
    # Missing history is NOT evidence of authenticity.
    if result.status == NO_MANIFEST or result.status == UNREADABLE:
        return MassFunction(
            m_auth=0.0,
            m_synth=0.0,
            m_uncert=1.0,
        )

    raise ValueError(f"unknown provenance status: {result.status}")


# Walks decoded manifest JSON and collects every digitalSourceType value.
def find_digital_source_types(obj) -> list:
    """Recursively collect all digitalSourceType strings from decoded JSON.

    Returns a (possibly empty) list of strings.
    """
    found = []

    # Branch 1: a dict -> check each key/value pair.
    if isinstance(obj, dict):
        for key in obj:
            value = obj[key]
            if key == "digitalSourceType" and isinstance(value, str):
                found.append(value)
            else:
                deeper = find_digital_source_types(value)
                found.extend(deeper)
        return found

    # Branch 2: a list -> walk each item.
    if isinstance(obj, list):
        for item in obj:
            deeper = find_digital_source_types(item)
            found.extend(deeper)
        return found

    # Branch 3: anything else (string, number, ...) -> nothing to find.
    return found


# Decides whether any collected source type declares AI generation.
def has_ai_claim(source_types: list) -> bool:
    """True if any source type mentions trained algorithmic media."""
    for source_type in source_types:
        lowered = source_type.lower()
        if AI_SOURCE_MARKER in lowered:
            return True
    return False
