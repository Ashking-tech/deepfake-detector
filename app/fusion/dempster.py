"""Mixer: combines checker opinions into one fused opinion.

Math used: Dempster's rule. For two opinions m1 and m2, multiply every pair
of shares, sort each product by what the two sides agree on (both say real
-> real; one says "don't know" -> follow the other side; real x fake ->
nobody agrees -> conflict K), then divide the agreed shares by (1 - K).

High-conflict guard: if K is huge (> threshold, default 0.6), normalizing
would produce paradox answers (Zadeh paradox), so we use Yager's fallback:
the conflict is moved into "unsure" instead of being divided away, which
pushes the final verdict to Unknown (honest abstain).
"""

from .mass import MassFunction

# Above this conflict level, classic Dempster normalizing is unsafe, so we
# fall back to Yager (conflict -> unsure). 0.6 = "checkers mostly disagree".
YAGER_K_THRESHOLD = 0.6


# Mixes exactly two opinions into one.
# Returns (fused opinion, K conflict level).
# K near 0 = checkers agree. K near 1 = checkers contradict each other.
# If K > threshold: Yager fallback (no division; K added to unsure).
def combine(
    m1: MassFunction,
    m2: MassFunction,
    yager_threshold: float = YAGER_K_THRESHOLD,
) -> tuple:
    """Combine m1 and m2 with Dempster's rule plus Yager fallback.

    Every product below is one row of the intersection table.
    Returns (fused MassFunction, K float).
    """
    # Unpack the first opinion into 3 plain numbers.
    a1 = m1.m_auth
    s1 = m1.m_synth
    u1 = m1.m_uncert

    # Unpack the second opinion into 3 plain numbers.
    a2 = m2.m_auth
    s2 = m2.m_synth
    u2 = m2.m_uncert

    # --- Products that land on REAL (the sides agree, or unsure follows) ---
    # Both say REAL.
    real_real = a1 * a2
    # First says REAL, second shrugs -> follow the first.
    real_unsure = a1 * u2
    # First shrugs, second says REAL -> follow the second.
    unsure_real = u1 * a2
    # Total REAL share before normalizing.
    auth = real_real + real_unsure
    auth = auth + unsure_real

    # --- Products that land on FAKE ---
    # Both say FAKE.
    fake_fake = s1 * s2
    # First says FAKE, second shrugs -> follow the first.
    fake_unsure = s1 * u2
    # First shrugs, second says FAKE -> follow the second.
    unsure_fake = u1 * s2
    # Total FAKE share before normalizing.
    synth = fake_fake + fake_unsure
    synth = synth + unsure_fake

    # --- Product that lands on UNSURE ---
    # Both sides shrug.
    uncert = u1 * u2

    # --- Products that land on CONFLICT (REAL x FAKE, nobody agrees) ---
    # First says REAL, second says FAKE.
    real_fake = a1 * s2
    # First says FAKE, second says REAL.
    fake_real = s1 * a2
    # Total contradiction level.
    conflict = real_fake + fake_real

    # Yager fallback path: contradiction is too big to divide away.
    if conflict > yager_threshold:
        # Do NOT divide (that would invent false certainty).
        # Park the contradiction in "unsure" so the verdict becomes Unknown.
        unsure_total = uncert + conflict
        fused = MassFunction(
            m_auth=auth,
            m_synth=synth,
            m_uncert=unsure_total,
        )
        return fused, conflict

    # Normal path: share out the contradiction proportionally.
    norm = 1.0 - conflict

    norm_auth = auth / norm
    norm_synth = synth / norm
    norm_uncert = uncert / norm

    fused = MassFunction(
        m_auth=norm_auth,
        m_synth=norm_synth,
        m_uncert=norm_uncert,
    )
    return fused, conflict


# Mixes a list of opinions left to right: ((m1 + m2) + m3) ...
# Returns (final fused opinion, K of the LAST mix).
# The last K tells classify.py how much the final step disagreed.
def sequential_fuse(
    masses: list,
    yager_threshold: float = YAGER_K_THRESHOLD,
) -> tuple:
    """Fuse 2 or more opinions in order.

    Needs at least 2 opinions. Returns (fused MassFunction, last_K float).
    """
    if len(masses) < 2:
        raise ValueError("need at least 2 mass functions to fuse")

    # Start with the first opinion as the running result.
    fused = masses[0]

    # Conflict of the most recent mix (updated each round).
    last_k = 0.0

    # Mix in the remaining opinions one at a time.
    rest = masses[1:]
    for m in rest:
        fused, last_k = combine(
            fused,
            m,
            yager_threshold,
        )

    return fused, last_k


# Lower bound of "how fake": only the points directly saying FAKE.
def belief_synth(m: MassFunction) -> float:
    """Belief that the image is AI-made.

    Counts only direct FAKE support, ignores the unsure share.
    """
    result = m.m_synth
    return result


# Upper bound of "how fake": fake points + unsure points (unsure MIGHT be
# fake). The gap plausibility - belief = m_uncert = how unsure we are.
def plausibility_synth(m: MassFunction) -> float:
    """Plausibility that the image is AI-made.

    Counts FAKE support plus the unsure share (nothing ruling it out).
    """
    result = m.m_synth + m.m_uncert
    return result
