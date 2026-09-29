"""Finite-shoe Blackjack environment.

Same rules and 2-action interface as Gymnasium's Blackjack-v1 (0=stand, 1=hit;
dealer hits while sum<17; player-sum/dealer-upcard/usable-ace observation), but
deals without replacement from a shuffled multi-deck shoe instead of sampling
ranks with replacement -- which is what makes card counting meaningful here.
Adds a Hi-Lo running/true count as a 4th observation field, updated as cards
actually become visible (the dealer's hole card is untagged until the player
stands and the dealer turns it over).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.card_counting import hilo_tag

# One suit's 13 ranks: Ace=1, 2-9 at face value, T/J/Q/K=10.
RANK_PATTERN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def cmp(a, b):
    return float(a > b) - float(a < b)


def usable_ace(hand):
    return int(1 in hand and sum(hand) + 10 <= 21)


def sum_hand(hand):
    return sum(hand) + 10 if usable_ace(hand) else sum(hand)


def is_bust(hand):
    return sum_hand(hand) > 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    return sorted(hand) == [1, 10]


class ShoeBlackjackEnv:
    """A reset()/step() env like Gymnasium's, backed by a finite reshuffling shoe."""

    def __init__(self, num_decks=6, penetration=0.75, natural=True, seed=None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.natural = natural
        self._rng = random.Random(seed)
        self.shoe = []
        self.cursor = 0
        self.running_count = 0
        self.player = []
        self.dealer = []
        self._new_shoe()

    def _new_shoe(self):
        single_deck = RANK_PATTERN * 4  # 52 cards: 4 aces, 4x(2..9), 16 tens
        self.shoe = single_deck * self.num_decks
        self._rng.shuffle(self.shoe)
        self.cursor = 0
        self.running_count = 0

    def _draw(self):
        if self.cursor >= len(self.shoe):
            self._new_shoe()  # defensive: a single hand ran past the shoe's end
        card = self.shoe[self.cursor]
        self.cursor += 1
        return card

    def _tag(self, card):
        self.running_count += hilo_tag(card)

    @property
    def decks_remaining(self):
        return max((len(self.shoe) - self.cursor) / 52, 0.5)

    @property
    def true_count(self):
        return self.running_count / self.decks_remaining

    def _obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.true_count)

    def reset(self):
        if self.cursor / len(self.shoe) >= self.penetration:
            self._new_shoe()
        self.player = [self._draw(), self._draw()]
        self.dealer = [self._draw(), self._draw()]  # dealer[1] is the hole card: not tagged yet
        for c in (self.player[0], self.player[1], self.dealer[0]):
            self._tag(c)
        return self._obs(), {}

    def step(self, action):
        assert action in (0, 1)
        if action == 1:  # hit
            card = self._draw()
            self.player.append(card)
            self._tag(card)
            if is_bust(self.player):
                return self._obs(), -1.0, True, False, {}
            return self._obs(), 0.0, False, False, {}

        # stand: reveal the hole card, play out the dealer, resolve
        self._tag(self.dealer[1])
        while sum_hand(self.dealer) < 17:
            card = self._draw()
            self.dealer.append(card)
            self._tag(card)
        reward = cmp(score(self.player), score(self.dealer))
        if self.natural and is_natural(self.player) and reward == 1.0:
            reward = 1.5
        return self._obs(), reward, True, False, {}


if __name__ == "__main__":
    env = ShoeBlackjackEnv(num_decks=6, penetration=0.75, seed=42)

    assert len(env.shoe) == 312
    assert env.shoe.count(1) == 24, "expected 24 aces in a 6-deck shoe"
    assert env.shoe.count(10) == 96, "expected 96 ten-value cards in a 6-deck shoe"
    print("deck composition OK: 312 cards, 24 aces, 96 tens")

    obs, _ = env.reset()
    assert abs(env.running_count) <= 3  # only 3 cards tagged so far
    print("fresh shoe running_count is near zero:", env.running_count)

    # Simulate a few thousand hands with a simple "stand on player total >= 17" policy
    # and sanity-check outcome rates look like plausible blackjack numbers.
    wins = losses = pushes = 0
    n_hands = 5000
    for _ in range(n_hands):
        obs, _ = env.reset()
        done = False
        while not done:
            player_sum = obs[0]
            action = 0 if player_sum >= 17 else 1
            obs, reward, done, _, _ = env.step(action)
        wins += reward > 0
        losses += reward < 0
        pushes += reward == 0
    print(f"{n_hands} hands, stand>=17 policy: win={wins/n_hands:.1%} "
          f"loss={losses/n_hands:.1%} push={pushes/n_hands:.1%}")
    assert 0.35 < wins / n_hands < 0.50
    assert 0.40 < losses / n_hands < 0.55
    print("shoe_env self-test passed")
