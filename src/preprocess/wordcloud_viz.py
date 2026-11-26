"""Korean notice word cloud visualization with automatic Korean font discovery."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from wordcloud import WordCloud

from src.preprocess.stopwords import load_stopwords
from src.preprocess.text import clean_text, tokenize_okt

_CANDIDATE_FONT_PATHS: tuple[Path, ...] = (
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
)


def find_korean_font_path() -> str | None:
    """Locate an installed Korean TrueType font via Windows Fonts or koreanize_matplotlib."""
    for candidate in _CANDIDATE_FONT_PATHS:
        if candidate.is_file():
            return str(candidate)

    try:
        from koreanize_matplotlib import get_font_ttf_path

        font_path = Path(get_font_ttf_path())
        if font_path.is_file():
            return str(font_path)
    except Exception:
        return None

    return None


def generate_wordcloud(
    texts: Sequence[str],
    output_path: str | Path,
    stopwords: set[str] | None = None,
    width: int = 800,
    height: int = 450,
) -> Path:
    """Generate a Korean notice word cloud PNG from corpus texts and return its Path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    active_stopwords = load_stopwords() if stopwords is None else stopwords
    token_counts: Counter[str] = Counter()
    for text in texts:
        cleaned = clean_text(text)
        if cleaned:
            token_counts.update(tokenize_okt(cleaned, mode="nouns", stopwords=active_stopwords))

    if not token_counts:
        raise ValueError("No valid tokens extracted to generate word cloud.")

    wc = WordCloud(
        font_path=find_korean_font_path(),
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=0.9,
    ).generate_from_frequencies(dict(token_counts))
    wc.to_file(str(out))
    return out
