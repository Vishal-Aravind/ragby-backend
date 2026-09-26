def split_message(text: str, limit: int = 4096) -> list:
    """Split text into pieces of at most `limit` chars for channels with a
    per-message cap (WhatsApp, Telegram both 4096). Breaks at line
    boundaries, so a list row is never cut in half; only a single line
    longer than the limit is hard-cut."""
    text = text or ""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return [p for p in parts if p.strip()] or [text[:limit]]
