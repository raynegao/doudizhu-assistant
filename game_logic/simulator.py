"""
Monte-Carlo style simulator for Dou Dizhu win-rate estimation.
"""

from dataclasses import dataclass
from typing import List

from .cards import Card
from .state_parser import GameState


@dataclass
class SimulationConfig:
    """
    Simulation parameters for win-rate estimation.
    """

    num_samples: int = 200
    max_steps: int = 500


def simulate_round(state: GameState, config: SimulationConfig) -> bool:
    """
    Simulate a single game and return True if the hero wins.
    """

    pass


def estimate_win_rate(state: GameState, num_samples: int = 200) -> float:
    """
    Estimate the win rate of the current player using repeated simulations.
    """

    pass
