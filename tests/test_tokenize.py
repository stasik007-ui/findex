import inspect

from findex.tokenize import tokenize


def test_is_generator_function() -> None:
    assert inspect.isgeneratorfunction(tokenize)


def test_mixed_case_and_german_eszett() -> None:
    assert list(tokenize("Python STRASSE Stra\u00dfe")) == ["python", "strasse", "strasse"]


def test_cyrillic_and_apostrophes() -> None:
    text = "Привіт, світ! П'ять м’яких булок."
    assert list(tokenize(text)) == ["привіт", "світ", "п'ять", "м’яких", "булок"]


def test_unicode_combining_marks() -> None:
    composed = list(tokenize("caf\u00e9"))
    decomposed = list(tokenize("cafe\u0301"))
    assert composed == decomposed == ["caf\u00e9"]


def test_hyphens_and_digits() -> None:
    text = "HTTP-200 state-of-the-art v3_beta --dash--"
    assert list(tokenize(text)) == ["http-200", "state-of-the-art", "v3", "beta", "dash"]


def test_punctuation_only() -> None:
    assert list(tokenize("... !!! ??? — () []")) == []


def test_empty_string() -> None:
    assert list(tokenize("")) == []
