"""Split long text into overlapping chunks for embedding.

A line-for-line port of langchain's RecursiveCharacterTextSplitter with its
defaults (separators "\\n\\n", "\\n", " ", ""; separator kept at the start
of each piece; whitespace stripped), verified to produce identical chunks.
langchain_text_splitters alone cost ~76MB of RAM at import on a 512MB
instance, for this much code.
"""
import re

_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_keep_separator(text: str, separator: str) -> list:
    if separator:
        parts = re.split(f"({re.escape(separator)})", text)
        splits = [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
        if len(parts) % 2 == 0:
            splits += parts[-1:]
        splits = [parts[0], *splits]
    else:
        splits = list(text)
    return [s for s in splits if s]


def _merge(splits: list, chunk_size: int, chunk_overlap: int) -> list:
    # Pieces already carry their separator, so they're joined with "".
    docs, current, total = [], [], 0
    for piece in splits:
        length = len(piece)
        if total + length > chunk_size:
            if current:
                doc = "".join(current).strip()
                if doc:
                    docs.append(doc)
                # Drop from the front until what's left fits as overlap.
                while total > chunk_overlap or (total + length > chunk_size and total > 0):
                    total -= len(current[0])
                    current = current[1:]
        current.append(piece)
        total += length
    doc = "".join(current).strip()
    if doc:
        docs.append(doc)
    return docs


def _split(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> list:
    separator, rest = separators[-1], []
    for i, sep in enumerate(separators):
        if not sep:
            separator = sep
            break
        if sep in text:
            separator, rest = sep, separators[i + 1:]
            break

    final, good = [], []
    for s in _split_keep_separator(text, separator):
        if len(s) < chunk_size:
            good.append(s)
            continue
        if good:
            final.extend(_merge(good, chunk_size, chunk_overlap))
            good = []
        if rest:
            final.extend(_split(s, rest, chunk_size, chunk_overlap))
        else:
            final.append(s)
    if good:
        final.extend(_merge(good, chunk_size, chunk_overlap))
    return final


def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list:
    return _split(text or "", _SEPARATORS, chunk_size, chunk_overlap)
