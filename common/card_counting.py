"""Hi-Lo card counting: the tag table and a classic bet-spread schedule."""
import math


def hilo_tag(card_value: int) -> int:
    """Hi-Lo tag for a card. 2-6 are low (+1), 7-9 are neutral (0), 10/A are high (-1)."""
    if 2 <= card_value <= 6:
        return 1
    if card_value in (1, 10):
        return -1
    return 0


def bet_units(true_count: float) -> int:
    """Classic 1-8 unit spread keyed off the true count, floored to an integer."""
    tc = math.floor(true_count)
    if tc <= 1:
        return 1
    if tc == 2:
        return 2
    if tc == 3:
        return 4
    if tc == 4:
        return 6
    return 8


if __name__ == "__main__":
    assert [hilo_tag(v) for v in (1, 2, 6, 7, 9, 10)] == [-1, 1, 1, 0, 0, -1]
    assert [bet_units(tc) for tc in (-2, 0, 1, 1.9, 2, 3, 4, 5, 9)] == [1, 1, 1, 1, 2, 4, 6, 8, 8]
    print("card_counting self-test passed")
