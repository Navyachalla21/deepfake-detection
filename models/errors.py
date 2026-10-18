class MediaError(Exception):
    """The uploaded file is unreadable or unsuitable (shown to the user as a 400)."""


class ModelNotTrainedError(Exception):
    """No trained weights found (shown to the user as a 503 with instructions)."""
