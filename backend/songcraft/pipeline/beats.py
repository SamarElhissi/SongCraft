from dataclasses import dataclass


@dataclass(frozen=True)
class LinePlacement:
    line: str
    start: float
    end: float


def map_lines_to_beats(
    lines: list[str],
    beat_times: list[float],
    duration: float,
) -> list[LinePlacement]:
    """Place each lyric line on [0, duration].

    An empty line list or a non-positive duration returns no placements, so
    callers never divide by zero. With no usable beats, or fewer gaps between
    beats than lines, the lines share the duration evenly. Otherwise lines are
    grouped across the beat grid and each one gets a positive-length window.
    """
    if not lines or duration <= 0:
        return []

    # Duplicate stamps would open a zero-length gap.
    interior = sorted({time for time in beat_times if 0 < time < duration})
    if not interior:
        return _even_slices(lines, duration)

    boundaries = [0.0, *interior, duration]
    gap_count = len(boundaries) - 1
    if gap_count < len(lines):
        # The prototype used beats // lines, which is 0 in this case.
        return _even_slices(lines, duration)

    base, extra = divmod(gap_count, len(lines))
    placements: list[LinePlacement] = []
    index = 0
    for line_index, line in enumerate(lines):
        width = base + (1 if line_index < extra else 0)
        start = boundaries[index]
        end = boundaries[index + width]
        placements.append(LinePlacement(line=line, start=start, end=end))
        index += width
    return placements


def _even_slices(lines: list[str], duration: float) -> list[LinePlacement]:
    step = duration / len(lines)
    return [
        LinePlacement(line=line, start=index * step, end=(index + 1) * step)
        for index, line in enumerate(lines)
    ]
