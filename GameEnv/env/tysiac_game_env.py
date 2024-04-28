import functools
import random
from typing import NamedTuple, Tuple, Any, Dict

import numpy as np

from pettingzoo import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector


class Card(NamedTuple):
    value: str
    suit: str


class TysiacGameEnv(AECEnv):
    metadata = {
        "name": "tysiac_game_env_v0",
    }

    def __init__(self):
        super().__init__()
        self.observations = None
        self._agent_selector = None
        self.widow_cards: list[Card] = []
        self.a_cards: list[Card] = []
        self.b_cards: list[Card] = []
        self.c_cards: list[Card] = []
        self.a_points = 0
        self.b_points = 0
        self.c_points = 0
        self.laid_cards: list[Card] = []
        self.current_turn_cards: list[tuple[str, Card]] = []
        self.possible_agents = ["a", "b", "c"]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.observations = {agent: None for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

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

    def step(self, action) -> tuple[Any, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict[str, Any]]]:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return self.observations, self.rewards, self.terminations, self.truncations, self.infos

        agent = self.agent_selection

        if agent == "a":
            self.laid_cards.append(self.a_cards.pop(action))

        if agent == "b":
            self.laid_cards.append(self.b_cards.pop(action))

        if agent == "c":
            self.laid_cards.append(self.c_cards.pop(action))

        self.rewards = self._get_empty_actors_dict(0)

        if len(self.current_turn_cards) == len(self.agents):
            winner = self._get_winner_for_current_turn()
            self.rewards[winner] = self._get_current_turn_points()
            winner_index = self.agents.index(winner)
            for _ in range(winner_index):
                agent = self.agents.pop(0)
                self.agents.append(agent)

        next_action_mask = np.zeros(7, dtype=np.int8)
        for i in range(len(self.a_cards) - 1):
            next_action_mask[i] = 1

        self.observations = self._get_empty_actors_dict({})
        self.observations["a"] = {
            "a_cards": self.a_cards,
            "laid_cards": self.laid_cards
        }
        self.observations["b"] = {
            "b_cards": self.b_cards,
            "laid_cards": self.laid_cards
        }
        self.observations["c"] = {
            "c_cards": self.c_cards,
            "laid_cards": self.laid_cards
        }

        self.terminations = self._get_empty_actors_dict(len(self.a_cards) == 0)
        self.truncations = self._get_empty_actors_dict(False)
        self.infos: dict = self._get_empty_actors_dict({})
        self.infos[agent]["action_mask"] = next_action_mask

        self.agent_selection = self._agent_selector.next()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _get_empty_actors_dict(self, initial_value):
        return {agent: initial_value for agent in self.agents}

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

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])
