"""Head-to-head comparison of all strategies in this repo, on matched
conditions, producing the repo's headline charts.

Rows:
  1. Simple_21's Q-table   -- flat bet,        infinite-deck, no count signal
  2. Basic strategy        -- flat bet,        6-deck shoe,   count-blind
  3. Hi-Lo counting        -- Hi-Lo bet spread, 6-deck shoe,   basic strategy + count-aware betting
  4. DQN                   -- Hi-Lo bet spread, 6-deck shoe,   learned count-aware play
  5. Hi-Lo, favorable rules -- Hi-Lo bet spread, 1-deck shoe + surrender, basic strategy

Rows 2 and 3 share a seed and an identical playing policy (basic strategy),
so they see identical cards and isolate exactly what the bet spread adds over
flat betting. Row 1 is evaluated under natural=True payouts (despite being
trained under natural=False) to match the other rows' payout rules -- always
standing on a made/natural 21 is optimal regardless of the payout multiplier,
so the trained policy is still valid there. Row 5 is the only row that
changes the *rules*, not just the strategy -- see rule_impact.py for the
full rule-by-rule breakdown of why. It's still a no-double/no-split 2-action
game, same as every other row; the only differences are deck count and
surrender, both real, independently adjustable rules in common/shoe_env.py.
"""
import numpy as np
import torch

from common.plot_style import STRATEGY_COLORS, INK_MUTED, apply_style, savefig  # sets Agg backend; import before pyplot

import matplotlib.pyplot as plt

from common.bankroll_sim import simulate
from common.basic_strategy import basic_strategy_action, should_surrender
from common.card_counting import bet_units
from common.dqn_model import BlackjackNet, normalize_obs
from common.shoe_env import ShoeBlackjackEnv

import gymnasium as gym

apply_style()

N_HANDS = 1_000_000
SEED = 555
Q_LEARNING = "Q-Learning (no count)"
BASIC_STRATEGY = "Basic Strategy (no count)"
HILO = "Hi-Lo Counting"
DQN = "DQN (count-aware)"
HILO_FAVORABLE = "Hi-Lo, favorable rules"
FLAT_BET = lambda tc: 1
SURRENDER_FN = lambda ps, du, ua, tc: should_surrender(ps, du, ua)


def load_qtable_policy():
    data = np.load("Simple_21\\checkpoints\\blackjack_brain.npy", allow_pickle=True).item()
    q_table = data["q_table"]

    def policy(player_sum, dealer_upcard, usable_ace, true_count):
        state = (player_sum, dealer_upcard, usable_ace)
        return int(np.argmax(q_table.get(state, [0.0, 0.0])))

    return policy


def basic_policy(player_sum, dealer_upcard, usable_ace, true_count):
    return basic_strategy_action(player_sum, dealer_upcard, usable_ace)


def load_dqn_policy():
    model = BlackjackNet(input_dim=4, output_dim=2)
    checkpoint = torch.load("NeuralNet_21\\checkpoints\\blackjack_dqn.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def policy(player_sum, dealer_upcard, usable_ace, true_count):
        state = torch.FloatTensor(normalize_obs(player_sum, dealer_upcard, usable_ace, true_count)).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(model(state)).item())

    return policy


def plot_bankroll_comparison(results, save_path):
    plt.figure(figsize=(9.5, 6))
    for name, stats in results.items():
        plt.plot(stats["bankroll_trajectory"], color=STRATEGY_COLORS[name], linewidth=1.6, label=name)
    plt.title("Bankroll Over Hands: All Five Strategies")
    plt.xlabel("Hand #")
    plt.ylabel("Bankroll (units)")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.6)
    savefig(save_path)


def plot_summary_bars(results, save_path):
    names = list(results.keys())
    edges = [results[n]["edge_pct"] for n in names]
    colors = [STRATEGY_COLORS[n] for n in names]

    plt.figure(figsize=(9.5, 5.5))
    bars = plt.bar(names, edges, color=colors, width=0.6)
    plt.axhline(0, color=INK_MUTED, linewidth=1)
    for bar, edge in zip(bars, edges):
        offset = 0.08 if edge >= 0 else -0.08
        va = "bottom" if edge >= 0 else "top"
        plt.text(bar.get_x() + bar.get_width() / 2, edge + offset, f"{edge:+.2f}%", ha="center", va=va)
    plt.title(f"Player Edge by Strategy ({N_HANDS:,} hands each)")
    plt.ylabel("Edge (% of amount wagered)")
    plt.xticks(rotation=12)
    plt.grid(axis="y", alpha=0.6)
    savefig(save_path)


def main():
    print(f"Running {N_HANDS:,}-hand matched comparison across 5 strategies...\n")

    results = {}

    qtable_policy = load_qtable_policy()
    env1 = gym.make("Blackjack-v1", natural=True, sab=False)
    results[Q_LEARNING] = simulate(env1, qtable_policy, FLAT_BET, N_HANDS)

    env2 = ShoeBlackjackEnv(seed=SEED)
    results[BASIC_STRATEGY] = simulate(env2, basic_policy, FLAT_BET, N_HANDS)

    env3 = ShoeBlackjackEnv(seed=SEED)
    results[HILO] = simulate(env3, basic_policy, bet_units, N_HANDS)

    dqn_policy = load_dqn_policy()
    env4 = ShoeBlackjackEnv(seed=SEED)
    results[DQN] = simulate(env4, dqn_policy, bet_units, N_HANDS)

    # Same strategy as row 3 (Hi-Lo), different rules: single deck + late
    # surrender, both real adjustable rules, not a different game. See
    # rule_impact.py for why this specific combination is what it takes to
    # get a no-double/no-split game to a genuine positive edge.
    env5 = ShoeBlackjackEnv(num_decks=1, seed=SEED)
    results[HILO_FAVORABLE] = simulate(env5, basic_policy, bet_units, N_HANDS, surrender_fn=SURRENDER_FN)

    print(f"{'Strategy':<28}{'Win%':>8}{'Push%':>8}{'Loss%':>8}{'Edge%':>10}{'Final Bankroll':>18}")
    for name, stats in results.items():
        print(f"{name:<28}{stats['win_rate']*100:>7.2f}%{stats['push_rate']*100:>7.2f}%"
              f"{stats['loss_rate']*100:>7.2f}%{stats['edge_pct']:>+9.2f}%{stats['final_bankroll']:>18,.0f}")

    plot_bankroll_comparison(results, "results\\compare_bankroll.png")
    plot_summary_bars(results, "results\\compare_summary.png")


if __name__ == "__main__":
    main()
