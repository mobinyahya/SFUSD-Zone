from dataclasses import dataclass


@dataclass
class Policy:
    name: str
    ctip: str
    tiebreaker: str
