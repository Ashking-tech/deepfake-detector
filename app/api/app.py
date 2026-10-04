"""Step 5 demo page: one upload button that runs all 3 checkers.

Plain story: you drop a photo in, it calls verify() from
orchestrate.py, and shows one answer card (final status + 3 checker
boxes + the unsure bar). Runs only on your own machine for now
(localhost, no public link).
"""

from app.api.orchestrate import log_result, verify

# Allowed types, shown next to the upload box.
ALLOWED_LABEL = "jpg / png / webp, max 10 MB"


# Turns a verify() dict into 3 short lines, one per checker.
def describe_pillars(result: dict) -> str:
    """Build human-readable per-pillar summary text."""
    # Step 1: grab the per-pillar block.
    per_pillar = result.get("per_pillar", {})

    # Step 2: one line per checker (missing entry -> "skipped").
    lines = []

    for name in ["provenance", "watermark", "forensics"]:
        entry = per_pillar.get(name)

        if entry is None:
            lines.append(f"{name}: skipped")

            continue

        detail = entry.get("detail", {})

        reason = detail.get("reason", "")

        line = f"{name}: synth={entry['m_synth']:.2f} ({reason})"

        lines.append(line)

    # Step 3: join the 3 lines.
    text = "\n".join(lines)

    return text


# Called by the Upload button: checks the file and returns the answer.
# Returns (status card text, pillar breakdown text, full JSON).
def analyze(image_path) -> tuple:
    """Gradio handler: verify upload, log it, return display strings."""
    # Step 1: no file yet (page just opened).
    if image_path is None or image_path == "":
        return "Upload an image to check.", "", {}

    # Step 2: run all 3 checkers.
    result = verify(image_path)

    # Step 3: save one log line (audit trail, best-effort).
    log_result(result, image_path)

    # Step 4: build the big answer card.
    masses = result["masses"]

    card = ""

    card += f"Verdict: {result['status']}\n"

    card += f"Driver: {result['driver']}\n"

    card += f"AI={masses['m_synth']:.2f} "

    card += f"Real={masses['m_auth']:.2f} "

    card += f"Unsure={masses['m_uncert']:.2f}\n"

    card += f"Range: [{result['belief']:.2f}, {result['plausibility']:.2f}] "

    card += f"Conflict K={result['K']:.2f}\n"

    card += f"Time: {result['latency_ms']:.0f} ms"

    # Step 5: build the per-checker breakdown.
    breakdown = describe_pillars(result)

    return card, breakdown, result


# Builds the Gradio page (imported lazily so tests stay gradio-free).
def build_demo():
    """Create the Gradio Blocks demo. Imports gradio only here."""
    import gradio as gr

    # Step 1: the page skeleton.
    with gr.Blocks(title="AI Media Authentication") as demo:
        # Title row.
        gr.Markdown("# AI Media Authentication (local demo)")

        gr.Markdown(f"Upload an image ({ALLOWED_LABEL}).")

        # Input row: the upload box.
        image = gr.Image(label="Upload image", type="filepath")

        # Button row.
        button = gr.Button("Check")

        # Output rows: answer card + checker breakdown + raw JSON.
        card = gr.Textbox(label="Verdict")

        breakdown = gr.Textbox(label="Per-checker breakdown")

        raw = gr.JSON(label="Full result")

        # Step 2: wire the button to analyze().
        button.click(
            fn=analyze,
            inputs=image,
            outputs=[card, breakdown, raw],
        )

    return demo


# Local launch only: localhost, no public share link.
def main():
    """Launch the demo on 127.0.0.1 (this machine only)."""
    # Step 1: build the page.
    demo = build_demo()

    # Step 2: serve it locally.
    demo.launch(
        server_name="127.0.0.1",
        share=False,
    )


if __name__ == "__main__":
    main()
