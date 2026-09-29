"""Optimal hit/stand basic strategy for this repo's 2-action ruleset (no double/split).

Rather than transcribe a textbook basic-strategy chart -- most published charts
assume double-down is available, and cells that say "double, else stand" vs
"double, else hit" don't collapse to a single rule you can safely guess from
memory -- this computes the exact optimal action from first principles via
expected-value recursion over an infinite-deck card distribution (the standard
assumption basic strategy is computed under), memoized into a lookup table.
"""
from collections import defaultdict
from functools import lru_cache

CARD_PROB = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}  # 1=Ace


def _transition(total, usable_ace, card):
    """Apply one more card to a (total, usable_ace) state, Gymnasium-style.

    total/usable_ace mirror Gymnasium's sum_hand()/usable_ace(): at most one
    ace is ever "promoted" to 11 (a second ace can never also be 11, since
    11+11 > 21), and once a promoted ace gets forced back down to 1 by a
    later card, it can never be re-promoted -- both facts fall out of always
    tracking hard_total (every ace counted as 1) and re-deriving the
    promotion from scratch each step, rather than carrying stale state.
    """
    hard_total = (total - 10 if usable_ace else total) + (1 if card == 1 else card)
    could_have_ace = usable_ace or (card == 1)
    if could_have_ace and hard_total + 10 <= 21:
        return hard_total + 10, True
    return hard_total, False


@lru_cache(maxsize=None)
def _dealer_dist(total, usable_ace):
    """Distribution over the dealer's final total, hitting while total<17. 0 = bust."""
    if total > 21:
        return {0: 1.0}
    if total >= 17:
        return {total: 1.0}
    dist = defaultdict(float)
    for card, p in CARD_PROB.items():
        nt, nu = _transition(total, usable_ace, card)
        for outcome, op in _dealer_dist(nt, nu).items():
            dist[outcome] += p * op
    return dict(dist)


def _dealer_start_dist(dealer_upcard):
    total, usable = _transition(0, False, dealer_upcard)
    return _dealer_dist(total, usable)


def _stand_ev(player_total, dealer_upcard):
    ev = 0.0
    for outcome, p in _dealer_start_dist(dealer_upcard).items():
        if outcome == 0 or outcome < player_total:
            ev += p * 1.0
        elif outcome > player_total:
            ev += p * -1.0
    return ev


def _hit_ev(total, usable_ace, dealer_upcard):
    return sum(
        p * _best_ev(*_transition(total, usable_ace, card), dealer_upcard)
        for card, p in CARD_PROB.items()
    )


@lru_cache(maxsize=None)
def _best_ev(total, usable_ace, dealer_upcard):
    if total > 21:
        return -1.0
    return max(_stand_ev(total, dealer_upcard), _hit_ev(total, usable_ace, dealer_upcard))


@lru_cache(maxsize=None)
def _optimal_action(total, usable_ace, dealer_upcard):
    if total > 21:
        return 0
    return 1 if _hit_ev(total, usable_ace, dealer_upcard) > _stand_ev(total, dealer_upcard) else 0


# Precompute a flat lookup table so hot simulation loops don't pay recursion cost.
_TABLE = {
    (player_sum, dealer_upcard, usable): _optimal_action(player_sum, usable, dealer_upcard)
    for player_sum in range(4, 22)
    for dealer_upcard in range(1, 11)
    for usable in (False, True)
}


def basic_strategy_action(player_sum, dealer_upcard, usable_ace, true_count=None):
    """Count-blind optimal hit(1)/stand(0) action. true_count is accepted and
    ignored -- keeps this usable as compare.py's count-blind baseline, and as
    the hook point for count-based index deviations if ever added later."""
    return _TABLE[(player_sum, dealer_upcard, bool(usable_ace))]


if __name__ == "__main__":
    for total in range(17, 22):
        for dealer in range(1, 11):
            assert basic_strategy_action(total, dealer, False) == 0, f"hard {total} vs {dealer} should stand"
    for total in range(4, 12):
        for dealer in range(1, 11):
            assert basic_strategy_action(total, dealer, False) == 1, f"hard {total} vs {dealer} should hit"
    print("basic_strategy invariant self-test passed (17-21 always stand, <=11 always hit)")

    print("\nHard totals (rows=player sum, cols=dealer upcard 1..10; H=hit S=stand):")
    print("     " + " ".join(f"{d:>2}" for d in range(1, 11)))
    for total in range(4, 22):
        row = " ".join(f"{'H' if basic_strategy_action(total, d, False) else 'S':>2}" for d in range(1, 11))
        print(f"{total:>3}: {row}")

    print("\nSoft totals (usable ace):")
    print("     " + " ".join(f"{d:>2}" for d in range(1, 11)))
    for total in range(12, 22):
        row = " ".join(f"{'H' if basic_strategy_action(total, d, True) else 'S':>2}" for d in range(1, 11))
        print(f"{total:>3}: {row}")
