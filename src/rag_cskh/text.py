"""Vietnamese text processing for BM25.

Vietnamese is written as space-separated syllables and most words have two syllables ("bảo hành").
Instead of a word segmenter, BM25 uses syllable unigrams + syllable bigrams (approximate compound words)
+ accent-stripped unigrams (customers often type without diacritics: "bao hanh").
Numbers are normalised: "15.000.000" -> "15000000".
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = set("""và là của có cho được không thì mà với các những một này đó khi để từ trong trên ra vào đã đang sẽ
rất cũng như nào gì bao nhiêu ạ ơi vậy nhé nha shop bên mình tôi tớ em anh chị ạ à ừ thế""".split())

_TOKEN = re.compile(r"[0-9a-zà-ỹđ]+(?:[.,][0-9]+)*", re.IGNORECASE)


def strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def syllables(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    out = []
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            tok = tok.replace(".", "").replace(",", "")
        out.append(tok)
    return out


def bm25_tokens(text: str) -> list[str]:
    syl = syllables(text)
    content = [s for s in syl if s not in STOPWORDS]
    toks = list(content)
    toks += [f"{a}_{b}" for a, b in zip(syl, syl[1:]) if not (a in STOPWORDS and b in STOPWORDS)]
    toks += ["~" + strip_accents(s) for s in content if strip_accents(s) != s]
    return toks
