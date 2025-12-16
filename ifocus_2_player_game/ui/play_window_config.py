from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PlayerConfig:
    """Configuration for a single player avatar in the play window."""

    number: int
    name: str
    wearing: bool = True


@dataclass
class SessionConfig:
    """Configuration for a play/training session shown in the window."""

    # "training" or "live"
    task_type: str
    players: List[PlayerConfig]
    # Total duration of a round in seconds
    duration_seconds: int = 60


def get_default_session_config() -> SessionConfig:
    """Return a default configuration.

    Edit this function (or create your own factory) to change
    how many players are shown, their names, and initial wearing
    status, without touching the UI logic.
    """

    return SessionConfig(
        task_type="training",  # or "live"
        players=[
            PlayerConfig(number=1, name="Player 1", wearing=True),
            PlayerConfig(number=2, name="Player 2", wearing=False),
        ],
        duration_seconds=60,
    )
