"""Shared opinion format for all three checkers.

Every checker (history file, hidden code, picture detective) must translate
its result into these same 3 numbers so the mixer in dempster.py can combine
them without knowing any checker details.
"""

from dataclasses import dataclass


@dataclass
class MassFunction:
    """One checker's opinion: 3 numbers that must add up to 1.0.

    m_auth:   points to REAL (authentic / camera-made).
    m_synth:  points to FAKE (AI-made / synthetic).
    m_uncert: honest "I don't know" share.

    Example: MassFunction(m_auth=0.10, m_synth=0.80, m_uncert=0.10) means
    "80% looks AI-made, 10% looks real, 10% unsure".
    """

    m_auth: float
    m_synth: float
    m_uncert: float

    # Runs automatically after MassFunction(...) is created.
    # Rejects garbage opinions so a buggy checker can never
    # silently poison the final answer.
    def __post_init__(self):
        # Check the REAL share is a valid probability.
        if self.m_auth < 0.0:
            raise ValueError(f"m_auth={self.m_auth} is negative")
        if self.m_auth > 1.0:
            raise ValueError(f"m_auth={self.m_auth} is above 1")

        # Check the FAKE share is a valid probability.
        if self.m_synth < 0.0:
            raise ValueError(f"m_synth={self.m_synth} is negative")
        if self.m_synth > 1.0:
            raise ValueError(f"m_synth={self.m_synth} is above 1")

        # Check the UNSURE share is a valid probability.
        if self.m_uncert < 0.0:
            raise ValueError(f"m_uncert={self.m_uncert} is negative")
        if self.m_uncert > 1.0:
            raise ValueError(f"m_uncert={self.m_uncert} is above 1")

        # Check the three shares add up to exactly 1.0.
        total = self.m_auth + self.m_synth
        total = total + self.m_uncert
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"masses sum to {total}, must sum to 1.0")


# Builds an "I know nothing" opinion: all weight on unsure.
# Used when e.g. the history file is missing (stripped by Instagram).
def vacuous() -> MassFunction:
    """Total ignorance.

    Mixing this with any other opinion leaves that opinion unchanged
    (it is the neutral element of the mixer).
    """
    result = MassFunction(
        m_auth=0.0,
        m_synth=0.0,
        m_uncert=1.0,
    )
    return result


# Shrinks a checker's confident shares toward "unsure" by rate alpha.
# alpha=0.1 means "I trust this checker 90%": 10% of its real/fake points
# move into unsure. Used to down-weight shaky sources (e.g. watermark with
# high bit-error rate) before mixing, so one overconfident checker cannot
# dictate the final answer.
def discount(m: MassFunction, alpha: float) -> MassFunction:
    """Discount opinion m by distrust rate alpha.

    alpha=0 keeps the opinion unchanged.
    alpha=1 turns it into fully unsure.
    Returns a brand new MassFunction, the input is left untouched.
    """
    # Alpha itself must be a valid rate.
    if alpha < 0.0:
        raise ValueError(f"alpha={alpha} is negative")
    if alpha > 1.0:
        raise ValueError(f"alpha={alpha} is above 1")

    # How much of the original opinion survives.
    keep = 1.0 - alpha

    # Shrink the decided shares.
    new_auth = m.m_auth * keep
    new_synth = m.m_synth * keep

    # Shrink the unsure share too, then add the distrusted part to it.
    new_uncert = m.m_uncert * keep
    new_uncert = new_uncert + alpha

    result = MassFunction(
        m_auth=new_auth,
        m_synth=new_synth,
        m_uncert=new_uncert,
    )
    return result
