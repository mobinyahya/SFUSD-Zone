from dataclasses import dataclass


@dataclass
class Policy:
    name: str
    ctip: str
    rounds_merged: str
    tiebreaker: str
