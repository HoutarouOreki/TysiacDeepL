import functools
import random
import numpy as np

from pettingzoo import ParallelEnv
from gymnasium import spaces


class TysiacGameEnv(ParallelEnv):
    metadata = {
        "name": "tysiac_game_env_v0",
    }

    def __init__(self):
        self.widow_cards: list[str] = []
        self.a_cards: list[str] = []
        self.b_cards: list[str] = []
        self.c_cards: list[str] = []
        self.a_points = 0
        self.b_points = 0
        self.c_points = 0
        self.laid_cards: list[str] = []
        self.current_turn_cards: list[tuple[str, str]] = []
        self.possible_agents = ["a", "b", "c"]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()

        self.a_cards = []
        self.b_cards = []
        self.c_cards = []
        self.laid_cards = []
        self.current_turn_cards = []

        card_values = "ATKQJ9"
        card_suits = "♠♥♦♣"
        deck = []
        picked_cards = [False for _ in range(24)]

        for card_suit in card_suits:
            for card_value in card_values:
                deck.append(card_suit + card_value)

        for agent_cards in (self.a_cards, self.b_cards, self.c_cards):
            for _ in range(7):
                random_card_index = -1
                while random_card_index == -1 or picked_cards[random_card_index]:
                    random_card_index = random.randint(0, 23)
                picked_cards[random_card_index] = True
                random_card = deck[random_card_index]
                agent_cards.append(random_card)

        self.widow_cards = [deck[x] for x in picked_cards if x is False]

    def step(self, actions):
        a_action = actions["a"]
        b_action = actions["b"]
        c_action = actions["c"]

        if a_action:
            self.laid_cards.append(self.a_cards.pop(a_action))

        if b_action:
            self.laid_cards.append(self.b_cards.pop(b_action))

        if c_action:
            self.laid_cards.append(self.c_cards.pop(c_action))

        if len(self.current_turn_cards) == len(self.agents):
            winner = self._get_winner_for_current_turn()
            winner_index = self.agents.index(winner)
            for _ in range(winner_index):
                agent = self.agents.pop(0)
                self.agents.append(agent)

    def _get_winner_for_current_turn(self) -> str:
        suit: str = self.current_turn_cards[0][1][0]

        highest_points: int = 0
        winner: str = self.current_turn_cards[0][0]
        for (agent, card) in self.current_turn_cards:
            agent: str
            card: str
            points = "9JQKTA".index(card[1]) if card[0] == suit else 0
            if points > highest_points:
                winner = agent

        return winner

    def _get_current_turn_points(self) -> int:
        points = 0
        for (agent, card) in self.current_turn_cards:
            if card[1] in "ATK":
                points += 10

        return points

    def render(self):
        pass

    def observation_space(self, agent):
        return spaces.Dict({
            f"{agent}_cards": spaces.Sequence(
                spaces.Tuple((
                    spaces.Text(1, charset="ATKQJ9"),
                    spaces.Text(1, charset="♠♥♦♣"))
                ),
            ),
            "laid_cards": spaces.Sequence(
                spaces.Tuple((
                    spaces.Text(1, charset="ATKQJ9"),
                    spaces.Text(1, charset="♠♥♦♣"))
                ),
            )
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(7)
