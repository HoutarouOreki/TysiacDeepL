import functools
import random
from typing import NamedTuple, Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class Card(NamedTuple):
    value: str
    suit: str

    def as_number(self) -> int:
        number = 0
        values = "ATKQJ9"
        suits = "♥♦♣♠"
        number += suits.index(self.suit) * len(values) + values.index(self.value)
        return number

    def __str__(self):
        return self.value + self.suit


def card_from_number(number) -> Card:
    values = "ATKQJ9"
    suits = "♥♦♣♠"
    value_number = number % len(values)
    suit_number = number // len(values)
    return Card(suit=suits[suit_number], value=values[value_number])


def _card_arr_to_str(cards: list[Card]) -> str:
    return " ".join([card.__str__() for card in cards])


class TysiacGameEnv(AECEnv):
    metadata = {
        "name": "tysiac_game_env_v0",
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        super().__init__()
        self.observations = None
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
        self.last_turn = {}
        self._agent_selector = agent_selector(self.possible_agents)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos: dict = {agent: {} for agent in self.agents}

        self.a_cards = []
        self.b_cards = []
        self.c_cards = []
        self.laid_cards = []
        self.current_turn_cards = []

        self.observations = self._get_empty_actors_dict({})
        self.observations["a"] = {
            "my_cards": np.array([x.as_number() for x in self.a_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }
        self.observations["b"] = {
            "my_cards": np.array([x.as_number() for x in self.b_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }
        self.observations["c"] = {
            "my_cards": np.array([x.as_number() for x in self.c_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }

        card_values = "ATKQJ9"
        card_suits = "♠♥♦♣"
        deck = []
        picked_cards = [False for _ in range(24)]

        for card_suit in card_suits:
            for card_value in card_values:
                deck.append(Card(suit=card_suit, value=card_value))

        for agent_cards in (self.a_cards, self.b_cards, self.c_cards):
            for _ in range(7):
                random_card_index = -1
                while random_card_index == -1 or picked_cards[random_card_index]:
                    random_card_index = random.randint(0, 23)
                picked_cards[random_card_index] = True
                random_card = deck[random_card_index]
                agent_cards.append(random_card)

        self.widow_cards = [deck[x] for x in picked_cards if x is False]

        self._set_observations()

    def step(self, action) -> tuple[Any, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict[str, Any]]]:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return self.observations, self.rewards, self.terminations, self.truncations, self.infos

        agent = self.agent_selection

        self._lay_card(agent, action)

        self.rewards = self._get_empty_actors_dict(0)

        if len(self.current_turn_cards) == len(self.agents):
            winner = self._get_winner_for_current_turn()
            points = self._get_current_turn_points()
            self.rewards[winner] = points
            self.last_turn = {
                "winner": winner,
                "points": points,
                "played_cards": self.current_turn_cards.copy()
            }
            self.infos["turn_completed"] = True
            self.current_turn_cards.clear()

            if winner == "a":
                self._agent_selector.reinit(["a", "b", "c"])
            elif winner == "b":
                self._agent_selector.reinit(["b", "c", "a"])
            elif winner == "c":
                self._agent_selector.reinit(["c", "a", "b"])
        else:
            self.infos["turn_completed"] = False

        self.agent_selection = self._agent_selector.next()

        next_action_mask = np.zeros(7, dtype=np.int8)
        for i in range(len(self.a_cards) - 1):
            next_action_mask[i] = 1

        self._set_observations()

        self.terminations = self._get_empty_actors_dict(len(self.a_cards) == 0)
        self.truncations = self._get_empty_actors_dict(False)
        self.infos[agent]["action_mask"] = next_action_mask

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _lay_card(self, agent, action):
        agent_cards = self.a_cards
        if agent == "b":
            agent_cards = self.b_cards
        elif agent == "c":
            agent_cards = self.c_cards

        card = agent_cards.pop(action)
        self.laid_cards.append(card)
        self.current_turn_cards.append((agent, card))

    def _set_observations(self):
        self.observations = self._get_empty_actors_dict({})
        self.observations["a"] = {
            "a_cards": np.array([x.as_number() for x in self.a_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }
        self.observations["b"] = {
            "b_cards": np.array([x.as_number() for x in self.b_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }
        self.observations["c"] = {
            "c_cards": np.array([x.as_number() for x in self.c_cards], dtype=np.int8),
            "laid_cards": np.array([x.as_number() for x in self.laid_cards], dtype=np.int8)
        }

    def _get_empty_actors_dict(self, initial_value):
        return {agent: initial_value for agent in self.agents}

    def _get_winner_for_current_turn(self) -> str:
        suit: str = self.current_turn_cards[0][1].suit

        highest_points: int = 0
        winner: str = self.current_turn_cards[0][0]
        for (agent, card) in self.current_turn_cards:
            agent: str
            card: Card
            points = "9JQKTA".index(card.value) if card.suit == suit else 0
            if points > highest_points:
                winner = agent

        return winner

    def _get_current_turn_points(self) -> int:
        points = 0
        for (agent, card) in self.current_turn_cards:
            if card.value in "ATK":
                points += 10

        return points

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) > 0:
            string = (f"Current state: \n\t"
                      f"a: {_card_arr_to_str(self.a_cards)}\n\t"
                      f"b: {_card_arr_to_str(self.b_cards)}\n\t"
                      f"c: {_card_arr_to_str(self.c_cards)}\n")
            if "turn_completed" in self.infos and self.infos["turn_completed"]:
                string += (f"Last turn: \n\t"
                           f"winner: {self.last_turn['winner']}\n\t"
                           f"points: {self.last_turn['points']}\n\t"
                           f"played cards: "
                           f"{' | '.join([f'{x[0]}: {x[1].__str__()}' for x in self.last_turn['played_cards']])}\n")
        else:
            string = "Game over"
        print(string)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            f"my_cards": spaces.Sequence(
                spaces.Discrete(24)
            ),
            "laid_cards": spaces.Sequence(
                spaces.Discrete(24)
            )
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(7)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up-to-date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observations[agent]
