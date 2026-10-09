"""Knowledge-base preprocessing: Markdown with front matter -> split by section (##) -> chunks.

Each chunk keeps document/section metadata plus a context header ("document title > section title") that is
prepended to the text used for BM25 and embeddings, so a short chunk such as "12 tháng" still says which
product's warranty it describes.

Chunk id: <doc_id>#<section_id>#<k>. Gold labels in the eval set point to <doc_id>#<section_id>.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings

HEADING_ID = re.compile(r"\s*\{#([a-z0-9\-]+)\}\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    status: str
    effective_date: str
    section_id: str
    section_title: str
    text: str

    @property
    def section_key(self) -> str:
        return f"{self.doc_id}#{self.section_id}"

    @property
    def search_text(self) -> str:
        expired = " (TÀI LIỆU ĐÃ HẾT HIỆU LỰC)" if self.status != "active" else ""
        return f"{self.title}{expired} > {self.section_title}\n{self.text}"


def normalize(text: str) -> str:
    """Unicode NFC (Vietnamese diacritics have several encodings) and whitespace normalisation."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slugify(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:60] or "muc"


def parse_document(path: Path) -> tuple[dict, list[tuple[str, str, str]]]:
    """-> (metadata, [(section_id, section_title, body)])."""
    raw = normalize(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if raw.startswith("---"):
        _, fm, raw = raw.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    meta.setdefault("doc_id", path.stem)
    meta.setdefault("title", path.stem)
    meta.setdefault("status", "active")
    sections, cur_id, cur_title, buf = [], "gioi-thieu", meta["title"], []
    for line in raw.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            if "\n".join(buf).strip():
                sections.append((cur_id, cur_title, "\n".join(buf).strip()))
            heading = line[3:].strip()
            m = HEADING_ID.search(heading)
            cur_title = HEADING_ID.sub("", heading).strip()
            cur_id = m.group(1) if m else slugify(cur_title)
            buf = []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        sections.append((cur_id, cur_title, "\n".join(buf).strip()))
    return meta, sections


def split_section(body: str, max_chars: int) -> list[str]:
    """Merge consecutive paragraphs / bullets up to about max_chars; split overly long paragraphs by sentence."""
    units = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        sentences = re.split(r"(?<=[.;:])\s+(?=[A-ZĐÀ-Ỹ0-9\-])", para) if "\n- " not in para else para.split("\n")
        units.extend(s.strip() for s in sentences if s.strip())
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def load_chunks(kb_dir: Path | None = None, max_chars: int | None = None) -> list[Chunk]:
    kb_dir, max_chars = kb_dir or settings.kb_dir, max_chars or settings.chunk_chars
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, sections = parse_document(path)
        for sid, stitle, body in sections:
            for k, text in enumerate(split_section(body, max_chars)):
                chunks.append(Chunk(chunk_id=f"{meta['doc_id']}#{sid}#{k}", doc_id=meta["doc_id"], title=meta["title"],
                                    category=meta.get("category", ""), status=meta.get("status", "active"),
                                    effective_date=meta.get("effective_date", ""), section_id=sid, section_title=stitle, text=text))
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids - check section anchors")
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
