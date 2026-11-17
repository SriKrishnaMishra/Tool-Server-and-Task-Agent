import pytest

from bodhan_mcp.languages import detect_script_language, is_supported, normalize_language


@pytest.mark.parametrize(
    ("text", "language"),
    [("வணக்கம்", "ta"), ("నమస్కారం", "te"), ("বাংলা", "bn"), ("hello", "en")],
)
def test_detect_script_language(text, language):
    result = detect_script_language(text)
    assert language in result["candidates"]


def test_devanagari_returns_ambiguous_candidates():
    result = detect_script_language("नमस्ते")
    assert result["language"] is None
    assert result["candidates"] == ["hi", "mr"]


def test_normalization_and_capability():
    assert normalize_language("Tamil") == "ta"
    assert is_supported("ta", "translate") is True
    assert is_supported("French", "translate") is False
