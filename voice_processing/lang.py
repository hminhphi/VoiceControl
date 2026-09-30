"""Lightweight script-based language detection for the voice pipeline.

Used to pick the TTS language when the STT backend does not report one, e.g.
sherpa-onnx SenseVoice running with ``STT_LANGUAGE=auto``. Detection is purely
Unicode-range based (no network / no extra dependency) and covers the languages
the stack speaks; anything unknown falls back to ``"en"``.
"""

_VN_SPECIFIC = set("ăâđêôơưĂÂĐÊÔƠƯ")


def _has_kana(text: str) -> bool:
    return any(
        0x3040 <= ord(c) <= 0x30FF or 0x31F0 <= ord(c) <= 0x31FF
        for c in text
    )


def _has_hangul(text: str) -> bool:
    return any(
        0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF
        for c in text
    )


def _has_han(text: str) -> bool:
    return any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF for c in text)


def _has_vietnamese(text: str) -> bool:
    # Vietnamese-only letters plus the Latin Extended Additional tone block.
    return any(c in _VN_SPECIFIC for c in text) or any(
        0x1EA0 <= ord(c) <= 0x1EFF for c in text
    )


def detect_lang(text: str | None) -> str | None:
    """Return a short language code (``ja``/``ko``/``vi``/``zh``/``en``) or None."""
    if not text or not text.strip():
        return None
    if _has_kana(text):
        return "ja"
    if _has_hangul(text):
        return "ko"
    if _has_vietnamese(text):
        return "vi"
    if _has_han(text):
        return "zh"
    return "en"
