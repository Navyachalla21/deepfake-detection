LOW_CONFIDENCE = 65.0  # below this the result is flagged as inconclusive


def label_from_prob(p_fake: float):
    """Return (label, confidence %) where confidence is the probability of the predicted class."""
    p_fake = float(min(max(p_fake, 0.0), 1.0))
    if p_fake >= 0.5:
        return "FAKE", round(p_fake * 100, 1)
    return "REAL", round((1 - p_fake) * 100, 1)


def low_confidence(conf: float) -> bool:
    return conf < LOW_CONFIDENCE


LOW_CONF_NOTE = " Confidence is low, so treat this as inconclusive rather than a verdict."
