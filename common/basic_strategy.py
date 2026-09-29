"""Optimal hit/stand basic strategy for this repo's 2-action ruleset (no double/split).

Rather than transcribe a textbook basic-strategy chart -- most published charts
assume double-down is available, and cells that say "double, else stand" vs
"double, else hit" don't collapse to a single rule you can safely guess from
memory -- this computes the exact optimal action from first principles via
expected-value recursion over an infinite-deck card distribution (the standard
assumption basic strategy is computed under), memoized into a lookup table.

Also computes the exact optimal late-surrender decision the same way: forfeit
half the bet is correct exactly when continuing (best of stand/hit EV) is
worse than -0.5.

Both the dealer-stand rule (S17, the default) and dealer-hits-soft-17 (H17)
are supported, parameterized through the same recursion -- see the
self-test's diff between the two tables for how much it actually changes
optimal play in this no-double ruleset (spoiler: barely at all, because H17
vs S17 mostly matters for double-down decisions this ruleset doesn't have).
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
def _dealer_dist(total, usable_ace, hit_soft_17):
    """Distribution over the dealer's final total. 0 = bust."""
    if total > 21:
        return {0: 1.0}
    must_hit = total < 17 or (hit_soft_17 and total == 17 and usable_ace)
    if not must_hit:
        return {total: 1.0}
    dist = defaultdict(float)
    for card, p in CARD_PROB.items():
        nt, nu = _transition(total, usable_ace, card)
        for outcome, op in _dealer_dist(nt, nu, hit_soft_17).items():
            dist[outcome] += p * op
    return dict(dist)


def _dealer_start_dist(dealer_upcard, hit_soft_17):
    total, usable = _transition(0, False, dealer_upcard)
    return _dealer_dist(total, usable, hit_soft_17)


def _stand_ev(player_total, dealer_upcard, hit_soft_17):
    ev = 0.0
    for outcome, p in _dealer_start_dist(dealer_upcard, hit_soft_17).items():
        if outcome == 0 or outcome < player_total:
            ev += p * 1.0
        elif outcome > player_total:
            ev += p * -1.0
    return ev


def _hit_ev(total, usable_ace, dealer_upcard, hit_soft_17):
    return sum(
        p * _best_ev(*_transition(total, usable_ace, card), dealer_upcard, hit_soft_17)
        for card, p in CARD_PROB.items()
    )


@lru_cache(maxsize=None)
def _best_ev(total, usable_ace, dealer_upcard, hit_soft_17):
    """Best EV from this state onward, NOT including the option to surrender
    (surrender is a decision made before this recursion even starts)."""
    if total > 21:
        return -1.0
    return max(_stand_ev(total, dealer_upcard, hit_soft_17),
                _hit_ev(total, usable_ace, dealer_upcard, hit_soft_17))


@lru_cache(maxsize=None)
def _optimal_action(total, usable_ace, dealer_upcard, hit_soft_17):
    if total > 21:
        return 0
    h = _hit_ev(total, usable_ace, dealer_upcard, hit_soft_17)
    s = _stand_ev(total, dealer_upcard, hit_soft_17)
    return 1 if h > s else 0


def _build_table(hit_soft_17):
    return {
        (player_sum, dealer_upcard, usable): _optimal_action(player_sum, usable, dealer_upcard, hit_soft_17)
        for player_sum in range(4, 22)
        for dealer_upcard in range(1, 11)
        for usable in (False, True)
    }


def _build_surrender_table(hit_soft_17):
    # Surrender guarantees exactly -0.5 EV; correct exactly when the best of
    # stand/hit is worse than that. Only ever relevant pre-decision on the
    # initial 2-card hand, but the table covers the full range harmlessly.
    return {
        (player_sum, dealer_upcard, usable): _best_ev(player_sum, usable, dealer_upcard, hit_soft_17) < -0.5
        for player_sum in range(4, 22)
        for dealer_upcard in range(1, 11)
        for usable in (False, True)
    }


# Precompute flat lookup tables so hot simulation loops don't pay recursion cost.
_TABLE_S17 = _build_table(hit_soft_17=False)
_TABLE_H17 = _build_table(hit_soft_17=True)
_SURRENDER_S17 = _build_surrender_table(hit_soft_17=False)
_SURRENDER_H17 = _build_surrender_table(hit_soft_17=True)


def basic_strategy_action(player_sum, dealer_upcard, usable_ace, true_count=None, dealer_hits_soft_17=False):
    """Count-blind optimal hit(1)/stand(0) action. true_count is accepted and
    ignored -- keeps this usable as compare.py's count-blind baseline, and as
    the hook point for count-based index deviations if ever added later."""
    table = _TABLE_H17 if dealer_hits_soft_17 else _TABLE_S17
    return table[(player_sum, dealer_upcard, bool(usable_ace))]


def should_surrender(player_sum, dealer_upcard, usable_ace, dealer_hits_soft_17=False):
    """Exact optimal late-surrender decision on the initial 2-card hand."""
    table = _SURRENDER_H17 if dealer_hits_soft_17 else _SURRENDER_S17
    return table[(player_sum, dealer_upcard, bool(usable_ace))]


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

    print("\nH17 vs S17: cells where optimal hit/stand action differs (expect very few --")
    print("H17 mostly changes DOUBLE decisions, which don't exist in this 2-action ruleset):")
    diffs = 0
    for player_sum in range(4, 22):
        for dealer in range(1, 11):
            for usable in (False, True):
                s17 = _TABLE_S17[(player_sum, dealer, usable)]
                h17 = _TABLE_H17[(player_sum, dealer, usable)]
                if s17 != h17:
                    diffs += 1
                    kind = "soft" if usable else "hard"
                    print(f"  {kind} {player_sum} vs dealer {dealer}: S17={'H' if s17 else 'S'} H17={'H' if h17 else 'S'}")
    print(f"  ({diffs} of 360 cells differ)")

    print("\nSurrender (S17): cells where optimal is to surrender pre-decision:")
    surrender_cells = [k for k, v in _SURRENDER_S17.items() if v and 4 <= k[0] <= 21]
    for player_sum, dealer, usable in sorted(surrender_cells):
        kind = "soft" if usable else "hard"
        print(f"  {kind} {player_sum} vs dealer {dealer}")
    assert (16, 10, False) in dict(_SURRENDER_S17) and _SURRENDER_S17[(16, 10, False)], \
        "hard 16 vs dealer 10 is the textbook always-surrender cell"
    print("should_surrender includes the textbook hard-16-vs-10 cell: OK")
