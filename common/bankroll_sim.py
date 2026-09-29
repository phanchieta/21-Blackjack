"""Run any (policy_fn, bet_fn) pair over many hands and report bankroll/edge stats.

Works with either ShoeBlackjackEnv (4-tuple obs, has .true_count) or
Gymnasium's own Blackjack-v1 (3-tuple obs) -- observations are normalized to
a common 4-tuple so every policy_fn sees the same shape regardless of env.
"""


def simulate(env, policy_fn, bet_fn, n_hands, starting_bankroll=10_000.0, unit_size=1.0):
    """
    env: an object with reset() -> (obs, info) and step(action) -> (obs, reward, terminated, truncated, info)
    policy_fn: (player_sum, dealer_upcard, usable_ace, true_count) -> 0 (stand) or 1 (hit)
    bet_fn: (true_count) -> bet size in units; multiplied by unit_size for the actual wager.
            true_count reflects the shoe state *before* this hand's cards are dealt.
    """
    bankroll = starting_bankroll
    trajectory = [bankroll]
    wins = pushes = losses = 0
    total_wagered = 0.0

    for _ in range(n_hands):
        true_count_for_bet = getattr(env, "true_count", 0.0)
        bet = bet_fn(true_count_for_bet) * unit_size

        obs, _ = env.reset()
        if len(obs) == 3:
            obs = (*obs, 0.0)
        done = False
        reward = 0.0
        while not done:
            action = policy_fn(*obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if len(obs) == 3:
                obs = (*obs, 0.0)
            done = terminated or truncated

        bankroll += bet * reward
        total_wagered += bet
        trajectory.append(bankroll)
        wins += reward > 0
        losses += reward < 0
        pushes += reward == 0

    return {
        "bankroll_trajectory": trajectory,
        "win_rate": wins / n_hands,
        "push_rate": pushes / n_hands,
        "loss_rate": losses / n_hands,
        "edge_pct": (bankroll - starting_bankroll) / total_wagered * 100.0,
        "final_bankroll": bankroll,
        "total_wagered": total_wagered,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.basic_strategy import basic_strategy_action
    from common.shoe_env import ShoeBlackjackEnv

    env = ShoeBlackjackEnv(seed=7)
    stats = simulate(
        env,
        policy_fn=lambda ps, du, ua, tc: basic_strategy_action(ps, du, ua),
        bet_fn=lambda tc: 1,
        n_hands=100_000,
    )
    print(f"basic strategy, flat bet, 100k hands: win={stats['win_rate']:.1%} "
          f"push={stats['push_rate']:.1%} loss={stats['loss_rate']:.1%} "
          f"edge={stats['edge_pct']:.2f}%")
    assert 0.40 < stats["win_rate"] < 0.46
    assert 0.03 < stats["push_rate"] < 0.12
    print("bankroll_sim smoke test passed")
