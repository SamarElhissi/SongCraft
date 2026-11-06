from songcraft.pipeline.beats import LinePlacement, map_lines_to_beats


def test_no_lines_returns_empty() -> None:
    assert map_lines_to_beats([], [0.5, 1.0], duration=8) == []


def test_non_positive_duration_returns_empty() -> None:
    assert map_lines_to_beats(["only line"], [0.5], duration=0) == []
    assert map_lines_to_beats(["only line"], [0.5], duration=-3) == []


def test_no_beats_splits_the_duration_evenly() -> None:
    placements = map_lines_to_beats(["one", "two"], [], duration=8)

    assert placements == [
        LinePlacement("one", 0, 4),
        LinePlacement("two", 4, 8),
    ]


def test_beats_outside_the_song_are_ignored() -> None:
    placements = map_lines_to_beats(["one", "two"], [0, -1, 12], duration=10)

    assert placements == [
        LinePlacement("one", 0, 5),
        LinePlacement("two", 5, 10),
    ]


def test_fewer_beats_than_lines_shares_the_timeline() -> None:
    placements = map_lines_to_beats(
        ["one", "two", "three", "four"],
        [1, 9],
        duration=10,
    )

    assert placements == [
        LinePlacement("one", 0, 2.5),
        LinePlacement("two", 2.5, 5),
        LinePlacement("three", 5, 7.5),
        LinePlacement("four", 7.5, 10),
    ]
    assert all(item.end > item.start for item in placements)


def test_extra_beats_are_grouped_across_lines() -> None:
    placements = map_lines_to_beats(["one", "two"], [2, 4, 6, 8], duration=10)

    assert placements == [
        LinePlacement("one", 0, 6),
        LinePlacement("two", 6, 10),
    ]


def test_duplicate_beats_do_not_create_an_empty_window() -> None:
    placements = map_lines_to_beats(["one", "two"], [2, 2, 4], duration=10)

    assert placements == [
        LinePlacement("one", 0, 4),
        LinePlacement("two", 4, 10),
    ]
    assert all(item.end > item.start for item in placements)


def test_single_line_covers_the_whole_song() -> None:
    placements = map_lines_to_beats(["only"], [1, 2, 3], duration=9)

    assert placements == [LinePlacement("only", 0, 9)]
