
import re
from typing import Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Module 1 - Session 1 API", version="1.0.0")


# -----------------------------
# Schemas (Pydantic models)
# -----------------------------

class NormalizeTextRequest(BaseModel):
    text: str = Field(..., min_length=20)
    lowercase: bool = False


class NormalizeTextResponse(BaseModel):
    normalized_text: str
    char_count: int
    word_count: int


class KeywordsRequest(BaseModel):
    text: str
    top_k: int = 5


class KeywordsResponse(BaseModel):
    keywords: list[str]


# -----------------------------
# Services (business logic)
# -----------------------------

def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(text: str, lowercase: bool = False) -> Tuple[str, int, int]:
    """Deterministic text normalization.

    Design notes:
    - Keep this function pure-ish (no global state, no I/O).
    - This style is easy to test and safe to reuse in pipelines.
    """
    if lowercase:
        text = text.lower()

    normalized = _collapse_whitespace(text)
    char_count = len(normalized)
    word_count = 0 if not normalized else len(normalized.split(" "))
    return normalized, char_count, word_count


def extract_keywords_simple(text: str, top_k: int) -> list[str]:
    """Simple deterministic keyword extraction.

    Steps:
    1. Lowercase
    2. Remove punctuation characters
    3. Split on whitespace
    4. Count frequencies
    5. Sort by frequency (desc) then alphabetically
    6. Return top_k tokens (ties deterministic via sort order)
    """
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    tokens = _collapse_whitespace(cleaned).split(" ")
    freq: dict[str, int] = {}
    for t in tokens:
        if not t:
            continue
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in sorted_tokens[:top_k]]


# -----------------------------
# Routes (API boundary)
# -----------------------------

@app.post("/normalize-text", response_model=NormalizeTextResponse)
def post_normalize_text(payload: NormalizeTextRequest) -> NormalizeTextResponse:
    normalized, char_count, word_count = normalize_text(
        text=payload.text,
        lowercase=payload.lowercase,
    )
    return NormalizeTextResponse(
        normalized_text=normalized,
        char_count=char_count,
        word_count=word_count,
    )


@app.post("/keywords-simple", response_model=KeywordsResponse)
def post_keywords_simple(payload: KeywordsRequest) -> KeywordsResponse:
    # business-rule validation (outside Pydantic schema)
    if payload.top_k < 1 or payload.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    keywords = extract_keywords_simple(payload.text, payload.top_k)
    return KeywordsResponse(keywords=keywords)
