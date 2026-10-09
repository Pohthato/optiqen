# worker/simulation/rally.py
"""A singles rally as a sequence of physically simulated shots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.court_model import COURT_LENGTH_M, NET_POST_HEIGHT_M, NET_Y_M, SINGLES_WIDTH_M
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, STEP_S, simulate, solve_launch, time_at_height

# Conservative: the tape is 1.55 m at the posts and 1.524 m at the centre.
NET_CLEARANCE_M = NET_POST_HEIGHT_M
# The body stands behind and to the side of the racket contact point (near player; mirrored for far).
BODY_OFFSET_M = (-0.3, 0.5)
# A receiver plays the shuttle between just off the floor and a jumping overhead reach.
MIN_RECEIVE_HEIGHT_M = 0.05
MAX_RECEIVE_HEIGHT_M = 3.5
DEFAULT_BODY = {"near": (SINGLES_WIDTH_M / 2, 3.2), "far": (SINGLES_WIDTH_M / 2, COURT_LENGTH_M - 3.2)}


@dataclass(frozen=True)
class Shot:
    kind: str
    target: tuple[float, float]
    flight_time: float
    receive_height: float | None  # None only for the rally's last shot, which lands


@dataclass(frozen=True)
class Contact:
    time: float
    position: tuple[float, float, float]
    hitter: str  # "near" | "far"
    kind: str


@dataclass(frozen=True)
class Flight:
    start_time: float
    end_time: float
    p0: tuple[float, float, float]
    v0: tuple[float, float, float]


@dataclass(frozen=True)
class Rally:
    contacts: tuple[Contact, ...]
    flights: tuple[Flight, ...]
    landing: tuple[float, float]
    end_time: float
    terminal_velocity: float

    def shuttle_at(self, time: float) -> np.ndarray | None:
        for flight in self.flights:
            if flight.start_time <= time <= flight.end_time:
                return simulate(flight.p0, flight.v0, [time - flight.start_time], self.terminal_velocity)[0]
        return None


def _check_net(position: np.ndarray, v0: np.ndarray, duration: float, terminal_velocity: float, label: str) -> None:
    times = np.arange(0.0, duration + STEP_S, STEP_S)
    xyz = simulate(position, v0, times, terminal_velocity)
    side = np.sign(xyz[:, 1] - NET_Y_M)
    crossings = np.where(np.diff(side) != 0)[0]
    if not len(crossings):
        raise ValueError(f"{label} does not cross the net before it is played")
    i = int(crossings[0])
    fraction = (NET_Y_M - xyz[i, 1]) / (xyz[i + 1, 1] - xyz[i, 1])
    height = xyz[i, 2] + fraction * (xyz[i + 1, 2] - xyz[i, 2])
    if height <= NET_CLEARANCE_M:
        raise ValueError(f"{label} hits the net ({height:.2f} m at the net plane)")


def build_rally(
    serve_position: tuple[float, float, float],
    shots: list[Shot],
    start_time: float = 0.5,
    terminal_velocity: float = FEATHER_TERMINAL_VELOCITY,
) -> Rally:
    if not shots:
        raise ValueError("a rally needs at least one shot")
    if any(shot.receive_height is None for shot in shots[:-1]) or shots[-1].receive_height is not None:
        raise ValueError("every shot but the last needs a receive height; the last shot lands")
    position = np.asarray(serve_position, dtype=np.float64)
    hitter = "near" if position[1] < NET_Y_M else "far"
    time = start_time
    contacts: list[Contact] = []
    flights: list[Flight] = []
    for index, shot in enumerate(shots):
        label = f"shot {index} ({shot.kind})"
        if shot.receive_height is not None and not MIN_RECEIVE_HEIGHT_M <= shot.receive_height <= MAX_RECEIVE_HEIGHT_M:
            raise ValueError(
                f"{label} receive height {shot.receive_height} m is outside a player's reach "
                f"({MIN_RECEIVE_HEIGHT_M}–{MAX_RECEIVE_HEIGHT_M} m)"
            )
        target = np.array([shot.target[0], shot.target[1], 0.0])
        if (target[1] - NET_Y_M) * (position[1] - NET_Y_M) >= 0:
            raise ValueError(f"{label} does not cross the net: its target is on the hitter's side")
        v0 = solve_launch(position, target, shot.flight_time, terminal_velocity)
        if shot.receive_height is None:
            duration = shot.flight_time
        else:
            duration = time_at_height(position, v0, shot.receive_height, terminal_velocity, max_time=shot.flight_time)
            if duration is None:
                raise ValueError(f"{label} never comes down to the receive height {shot.receive_height} m")
        _check_net(position, v0, duration, terminal_velocity, label)
        contacts.append(Contact(time, tuple(float(c) for c in position), hitter, shot.kind))
        flights.append(Flight(time, time + duration, tuple(float(c) for c in position), tuple(float(c) for c in v0)))
        position = simulate(position, v0, [duration], terminal_velocity)[0]
        time += duration
        hitter = "far" if hitter == "near" else "near"
    return Rally(tuple(contacts), tuple(flights), (float(position[0]), float(position[1])), time, terminal_velocity)


def canned_rally() -> Rally:
    """A realistic club-level singles rally (~8 s): serve, clear, drop, lift, smash, block, net."""
    return build_rally(
        (3.3, 3.6, 1.0),
        [
            Shot("serve", (1.5, 12.9), 1.9, 2.4),
            Shot("clear", (3.5, 0.6), 1.8, 2.4),
            Shot("drop", (2.0, 8.3), 1.2, 0.5),
            Shot("lift", (1.2, 0.7), 1.7, 2.5),
            Shot("smash", (4.0, 11.6), 0.75, 1.0),
            Shot("block", (3.2, 5.0), 1.1, 0.6),
            Shot("net", (2.6, 7.8), 1.1, None),
        ],
    )


def player_positions(rally: Rally, time: float) -> dict[str, tuple[float, float]]:
    """Body centre of each player: at their contact points (offset behind the racket), moving linearly between them."""
    positions: dict[str, tuple[float, float]] = {}
    for side, sign in (("near", 1.0), ("far", -1.0)):
        own = [contact for contact in rally.contacts if contact.hitter == side]
        if not own:
            positions[side] = DEFAULT_BODY[side]
            continue
        times = [contact.time for contact in own]
        xs = [contact.position[0] + sign * BODY_OFFSET_M[0] for contact in own]
        ys = [contact.position[1] - sign * BODY_OFFSET_M[1] for contact in own]
        positions[side] = (float(np.interp(time, times, xs)), float(np.interp(time, times, ys)))
    return positions


# Shot families: (depth past the net in m, flight time in s, receive height in m, contact heights
# the shot can be played from). Depth is measured into the receiver's half.
SHOT_TYPES: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "clear": ((5.2, 6.5), (1.5, 2.1), (1.9, 2.8), (1.6, 3.5)),
    "drop": ((1.0, 2.6), (0.9, 1.4), (0.2, 1.0), (1.4, 3.5)),
    "smash": ((2.5, 5.0), (0.5, 0.9), (0.4, 1.5), (1.8, 3.5)),
    "drive": ((3.0, 5.5), (0.6, 1.0), (1.0, 1.7), (0.9, 2.0)),
    "push": ((2.5, 4.5), (0.7, 1.1), (0.8, 1.6), (0.4, 1.6)),
    "lift": ((5.0, 6.4), (1.4, 2.0), (1.9, 2.8), (0.05, 1.4)),
    "net": ((0.4, 1.6), (0.7, 1.1), (0.2, 0.8), (0.05, 1.3)),
    "block": ((0.8, 2.2), (0.8, 1.2), (0.2, 0.8), (0.05, 1.6)),
}
SERVES = {"low": ((2.0, 2.9), (0.9, 1.3), (0.4, 1.2)), "high": ((5.6, 6.5), (1.7, 2.2), (1.9, 2.8))}


def _try_shot(position: np.ndarray, kind: str, target: np.ndarray, flight_time: float, receive: float | None, terminal_velocity: float):
    """The next contact position, or None when this shot is not physically possible from here."""
    try:
        v0 = solve_launch(position, target, flight_time, terminal_velocity)
        duration = flight_time if receive is None else time_at_height(position, v0, receive, terminal_velocity, max_time=flight_time)
        if duration is None:
            return None
        _check_net(position, v0, duration, terminal_velocity, kind)
    except ValueError:
        return None
    return simulate(position, v0, [duration], terminal_velocity)[0]


def random_rally(rng: np.random.Generator, terminal_velocity: float = FEATHER_TERMINAL_VELOCITY) -> Rally:
    """A random, physically valid singles rally: a low or high serve from the near half, then
    2-10 shots, each chosen from those playable at its contact height (smashes and clears from
    overhead, lifts and net shots from low), the last one landing in court."""
    count = int(rng.integers(2, 11))
    right = bool(rng.random() < 0.5)
    serve_position = np.array([rng.uniform(3.0, 4.6) if right else rng.uniform(0.6, 2.2), rng.uniform(2.8, 4.4), rng.uniform(0.9, 1.15)])
    for _ in range(200):
        position, near, shots = serve_position, True, []
        for index in range(count):
            last = index == count - 1
            for _attempt in range(40):
                if index == 0:
                    depth, flight, receive = SERVES["high" if rng.random() < 0.5 else "low"]
                    kind = "serve"
                    x = rng.uniform(0.3, 2.4) if right else rng.uniform(2.8, 4.9)  # diagonally opposite service court
                else:
                    playable = [name for name, (*_, reach) in SHOT_TYPES.items() if reach[0] <= position[2] <= reach[1]]
                    kind = str(rng.choice(playable))
                    depth, flight, receive, _ = SHOT_TYPES[kind]
                    x = rng.uniform(0.3, 4.9)
                d = rng.uniform(*depth)
                target = np.array([x, NET_Y_M + d if near else NET_Y_M - d, 0.0])
                flight_time = float(rng.uniform(*flight))
                height = None if last else float(rng.uniform(*receive))
                landed = _try_shot(position, kind, target, flight_time, height, terminal_velocity)
                if landed is not None:
                    shots.append(Shot(kind, (float(target[0]), float(target[1])), flight_time, height))
                    position, near = landed, not near
                    break
            else:
                break
        if len(shots) == count:
            return build_rally(tuple(float(c) for c in serve_position), shots, terminal_velocity=terminal_velocity)
    raise RuntimeError("could not build a random rally")
