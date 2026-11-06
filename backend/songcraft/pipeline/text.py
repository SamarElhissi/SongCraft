import re

_LABEL = re.compile(
    r"^(?:verse|chorus|bridge|intro|outro|hook|pre-chorus|refrain)\b[\s\d:.\-]*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"^\d+\s*[\).:\-]\s*")
_BULLET = re.compile(r"^[-*•]\s+")
_COMMENTARY = re.compile(r"^(?:here|sure|okay|lyrics)\b", re.IGNORECASE)


def parse_lyric_lines(text: str) -> list[str]:
    """Keep one lyric per line and drop section labels the model sometimes adds."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip('"').strip("*").strip()
        line = _BULLET.sub("", line)
        line = _LABEL.sub("", line).strip()
        line = _NUMBER.sub("", line).strip(" -")
        if not line:
            continue
        if line.endswith(":") and len(line.split()) <= 4:
            continue
        if _COMMENTARY.match(line) and len(line.split()) <= 8:
            continue
        lines.append(line)
    return lines


def split_lyrics_into_lines(lyrics: str, max_line_length: int = 50) -> list[str]:
    """Split on punctuation, then fold any remaining long line at a word boundary."""
    lyric_lines: list[str] = []
    for raw_line in re.split(r"[.,;!?]", lyrics):
        line = raw_line.strip()
        while len(line) > max_line_length:
            split_index = line[:max_line_length].rfind(" ")
            if split_index <= 0:
                split_index = max_line_length
            piece = line[:split_index].strip()
            if piece:
                lyric_lines.append(piece)
            line = line[split_index:].strip()
        if line:
            lyric_lines.append(line)
    return lyric_lines
