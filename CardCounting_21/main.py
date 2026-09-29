"""Classic card counting: basic strategy for playing decisions + a Hi-Lo
true-count bet spread for sizing. No training, no neural net -- this is the
rule-based baseline the ML agents in this repo get compared against.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bankroll_sim import simulate
from common.basic_strategy import basic_strategy_action
from common.card_counting import bet_units
from common.shoe_env import ShoeBlackjackEnv

from visualize import plot_bankroll, plot_edge_vs_true_count, plot_true_count_distribution


def policy_fn(player_sum, dealer_upcard, usable_ace, true_count):
    return basic_strategy_action(player_sum, dealer_upcard, usable_ace)


def edge_vs_true_count(n_hands):
    """Re-simulate, bucketing each hand's outcome by the true count it was
    dealt at, to trace out how the player's edge moves with the count."""
    env = ShoeBlackjackEnv(seed=202)
    buckets = defaultdict(lambda: [0.0, 0])  # true_count bucket -> [reward sum, n]
    all_counts = []

    for _ in range(n_hands):
        tc_bucket = round(env.true_count)
        all_counts.append(env.true_count)
        obs, _ = env.reset()
        done = False
        reward = 0.0
        while not done:
            action = basic_strategy_action(*obs[:3])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        buckets[tc_bucket][0] += reward
        buckets[tc_bucket][1] += 1

    edge_by_count = {tc: s / n for tc, (s, n) in buckets.items() if n >= 30}
    return edge_by_count, all_counts


def main():
    n_hands = 500_000

    print(f"Simulating {n_hands:,} hands each, matched shoes (same seed -> identical cards, "
          f"since both use the same count-blind playing policy and differ only in bet size)...")

    # Same seed for both: bet size is the only thing that differs, so the
    # comparison isolates the bet-spread's contribution with zero card-luck variance.
    hilo_stats = simulate(ShoeBlackjackEnv(seed=101), policy_fn, bet_units, n_hands)
    flat_stats = simulate(ShoeBlackjackEnv(seed=101), policy_fn, lambda tc: 1, n_hands)

    print("\n--- Hi-Lo spread betting ---")
    print(f"Win {hilo_stats['win_rate']:.1%}  Push {hilo_stats['push_rate']:.1%}  Loss {hilo_stats['loss_rate']:.1%}")
    print(f"Edge: {hilo_stats['edge_pct']:+.2f}%   Final bankroll: {hilo_stats['final_bankroll']:,.0f} "
          f"(started at 10,000, 1 unit = 1 bankroll unit)")

    print("\n--- Flat betting (no counting, same cards/decisions) ---")
    print(f"Win {flat_stats['win_rate']:.1%}  Push {flat_stats['push_rate']:.1%}  Loss {flat_stats['loss_rate']:.1%}")
    print(f"Edge: {flat_stats['edge_pct']:+.2f}%   Final bankroll: {flat_stats['final_bankroll']:,.0f}")

    print("\nComputing player edge vs. true count...")
    edge_by_count, all_counts = edge_vs_true_count(n_hands)

    plot_bankroll(hilo_stats["bankroll_trajectory"], flat_stats["bankroll_trajectory"],
                  save_path="results\\bankroll_trajectory.png")
    plot_true_count_distribution(all_counts, save_path="results\\true_count_distribution.png")
    plot_edge_vs_true_count(edge_by_count, save_path="results\\edge_vs_true_count.png")


if __name__ == "__main__":
    main()
