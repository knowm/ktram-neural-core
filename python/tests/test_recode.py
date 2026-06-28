"""recode/ — pure functions over the lane-y vector."""

from ktram_neural_core.recode import AboveZero, Winner, WinnerAboveZero


def test_winner_picks_argmax():
    assert Winner().recode([-0.2, 0.7, 0.3]) == (1,)
    assert Winner().recode([-0.5, -0.9, -0.1]) == (2,)   # argmax even when all < 0


def test_above_zero_returns_positive_channels():
    assert AboveZero().recode([-0.2, 0.7, 0.3, -0.1]) == (1, 2)
    assert AboveZero().recode([-0.2, -0.7]) == ()


def test_winner_above_zero_abstains_when_best_not_positive():
    assert WinnerAboveZero().recode([-0.2, 0.7, 0.3]) == (1,)
    assert WinnerAboveZero().recode([-0.5, -0.9, -0.1]) == ()
    assert WinnerAboveZero().recode([]) == ()
