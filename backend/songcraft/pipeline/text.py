import re


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
