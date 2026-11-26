import reimport unicodedatafrom collections.abc import Iterator# Зберігаємо слова та числа, включно зі словами з внутрішніми апострофами або дефісами.
# Символи підкреслення (_) та крайня пунктуація ігноруються.
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Лінивий токенізатор: NFC-нормалізація -> casefold -> re.finditer."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group(0)