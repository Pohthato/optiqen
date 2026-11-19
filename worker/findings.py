# worker/findings.py
"""What a rally's analysis found, as the findings coaching may phrase, and nothing else (spec
section 2). Each has an ID, the player it is about, the shots it concerns, the frames it was seen
in, its facts, its call and one plain sentence that says it. A call is made with its chance or
left too close to call, as the analysis left it, and a measured number keeps its standard
deviation. Findings are made from an analysis as analyse_rally.py writes it, so a saved analysis
gets them without its video; each frame's time comes from the shuttle detections when they are
given, and from the frame rate when not.

The kinds:
- "end": how the last shot came down (in, out or into the net) and who won the point;
- "how": how the rally was won: a winner, let in, or a forced or unforced error
  (rally_reasoning.py);
- "turn": the first of the winner's shots from which every one of theirs left the loser under
  pressure (rally_reasoning.py), and "valueTurn": where the rally turned by each player's chance
  of winning it (rally_value.py);
- "shot": one shot: its name and details, speed, height, how far over the net, where it came
  down, how hard it made the receiver work, the hitter's body at the hit, where both players
  stood, and each player's chance of winning from it;
- "player": a player's reach and racket arm.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Any, Sequence

from shots import CALL_CONFIDENCE

TOO_CLOSE = "too close to call"
OTHER = {"near": "far", "far": "near"}
# The hitter's body at the hit and their readiness before it, as analyse_rally reports them.
BODY = ("aboveReachM", "elbowDeg", "sideOnDeg", "frontKneeDeg", "offFloorM", "weightAheadM", "hipsHeightM")
READY = ("fromBaseM", "splitDropM", "splitLowestMs")


def findings(analysis: dict[str, Any], times_ms: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """The rally's findings, numbered F1, F2, ... in the order: end, how, turns, shots, players.
    `times_ms` is every frame's time, as the shuttle detections give it."""
    shots, summary = analysis.get("shots") or [], analysis.get("summary") or {}
    out: list[dict[str, Any]] = []

    def frame_at(time_ms: float) -> int:
        """The first frame at or after a time."""
        if times_ms is None:
            return int(math.ceil(time_ms * analysis["fps"] / 1000.0 - 1e-6))
        return min(bisect_left(times_ms, time_ms - 1e-6), len(times_ms) - 1)

    def time_of(frame: int) -> float:
        return round(frame * 1000.0 / analysis["fps"] if times_ms is None else times_ms[frame], 1)

    contacts = [(event["timeMs"], event["frame"]) for event in analysis.get("events", []) if event["type"] == "contact"]

    def hit_frame(time_ms: float) -> int:
        """The frame a hit was first seen after, as the analysis's contacts give it."""
        nearest = min(contacts, key=lambda c: abs(c[0] - time_ms), default=None)
        return nearest[1] if nearest is not None and abs(nearest[0] - time_ms) < 1.0 else frame_at(time_ms)

    def add(kind: str, about: str | None, numbers: list[int], frames: list[int], call: str, p: float | None, facts: dict, text: str) -> None:
        evidence = sorted({frame for frame in frames if frame is not None})
        out.append({
            "id": f"F{len(out) + 1}", "kind": kind, "about": about, "shots": numbers,
            "evidence": [{"frame": frame, "timeMs": time_of(frame)} for frame in evidence],
            "call": call, "p": None if p is None else round(p, 3), "facts": facts, "text": text,
        })

    def frame_of(number: int) -> int:
        return hit_frame(shots[number - 1]["hitMs"])

    end = summary.get("end")
    if end is not None and shots:
        number, hitter = end["shot"], end["hitter"]
        landing = shots[number - 1].get("landing")
        frames = [frame_of(number), frame_at(end["timeMs"])]
        facts = {"p": end["p"], "winner": end["winner"], "pHitterWon": end["pHitterWon"]}
        if landing is not None:
            facts["landing"] = _point(landing["point"], landing["covariance"])
        where = {"in": "came down in", "out": "came down out", "net": "went into the net"}
        if end["call"] in where:
            won = f"so the {end['winner']} player won the point" if end["winner"] in OTHER else "and who won the point is too close to call"
            text = f"Shot {number}, the {hitter} player's, {where[end['call']]} ({_percent(end['p'][end['call']])}), {won}."
            add("end", hitter, [number], frames, end["call"], end["p"][end["call"]], facts, text)
        else:
            chances = ", ".join(f"{kind} {_percent(p)}" for kind, p in end["p"].items())
            add("end", hitter, [number], frames, TOO_CLOSE, None, facts, f"Where shot {number}, the {hitter} player's, came down is too close to call: {chances}.")

    reasons = summary.get("reasons")
    if reasons is not None and end is not None and shots:
        last, hitter = end["shot"], end["hitter"]
        call, p = reasons["call"], reasons["p"]
        error = call in ("forced error", "unforced error")
        numbers = [last - 1, last] if last > 1 and call in ("forced error", TOO_CLOSE) else [last]
        about = OTHER[hitter] if call == "let in" else hitter if call != TOO_CLOSE else None
        facts: dict[str, Any] = {"p": p}
        if error and last > 1 and shots[last - 2].get("pressure") is not None:
            facts["spareMs"] = shots[last - 2]["pressure"]["spareMs"]  # what the shot before left the one who erred
        if call == "winner":
            text = f"The {hitter} player won the rally with a winner, shot {last} ({_percent(p[call])})."
        elif call == "let in":
            text = f"The {about} player let shot {last} go, and it came down in ({_percent(p[call])})."
        elif error:
            kind = "forced: the shot before left them less than a quarter of a second to spare" if call == "forced error" else "unforced: the shot before left them time to spare"
            text = f"The {hitter} player's error on shot {last} was {kind} ({_percent(p[call])})."
        else:
            chances = ", ".join(f"{how} {_percent(q)}" for how, q in p.items() if q >= 0.005)
            text = f"How the rally was won is too close to call: {chances}."
        add("how", about, numbers, [frame_of(n) for n in numbers], call, p.get(call), facts, text)

    for kind, entry in (("turn", (reasons or {}).get("turn")), ("valueTurn", summary.get("valueTurn"))):
        if entry is None or not shots or end is None:
            continue
        lean = max(range(len(entry["pAt"])), key=lambda i: entry["pAt"][i]) if entry["pAt"] else None
        facts = {"pAt": {str(i + 1): p for i, p in enumerate(entry["pAt"]) if p >= 0.005}, "pNoTurn": entry["pNoTurn"]}
        by = "every one of their shots from it left the other player under pressure" if kind == "turn" else "their chance of winning stayed above even from it"
        prefix = "The rally" if kind == "turn" else "By each player's chance of winning, the rally"
        if isinstance(entry["call"], int):
            number = entry["call"]
            hitter = shots[number - 1]["hitter"]
            text = f"{prefix} turned at shot {number}, the {hitter} player's: {by} ({_percent(entry['pAt'][number - 1])})."
            add(kind, hitter, [number], [frame_of(number)], f"shot {number}", entry["pAt"][number - 1], facts, text)
        elif entry["call"] == "no turn":
            why = ": the winner's last shot left the other player time to spare" if kind == "turn" else ""
            text = f"{prefix} did not turn{why} ({_percent(entry['pNoTurn'])})."
            add(kind, None, [end["shot"]], [frame_of(end["shot"])], "no turn", entry["pNoTurn"], facts, text)
        else:
            leaning = "no turn" if lean is None or entry["pNoTurn"] >= entry["pAt"][lean] else f"shot {lean + 1}"
            p = entry["pNoTurn"] if leaning == "no turn" else entry["pAt"][lean]
            numbers = [end["shot"]] if leaning == "no turn" else [lean + 1]
            text = f"Where the rally turned{'' if kind == 'turn' else ' by the chances of winning'} is too close to call; it leans to {leaning} ({_percent(p)})."
            add(kind, None, numbers, [frame_of(n) for n in numbers], TOO_CLOSE, None, {**facts, "leans": leaning}, text)

    for shot in shots:
        number, hitter = shot["shot"], shot["hitter"]
        facts, said = _shot_facts(shot)
        frames = [frame_of(number)]
        if shot.get("landing") is not None:
            frames.append(frame_at(shot["landing"]["timeMs"]))
        parts = [f"Shot {number}, the {hitter} player's: {said}"]
        if "speedMS" in facts and facts["speedMS"]["sigma"] < facts["speedMS"]["value"] / 2:
            parts.append(f"{facts['speedMS']['value']:.1f} ± {facts['speedMS']['sigma']:.1f} m/s off the racket")
        if "netClearanceM" in facts and facts["netClearanceM"]["sigma"] < 0.25:
            parts.append(f"{facts['netClearanceM']['value']:.2f} ± {facts['netClearanceM']['sigma']:.2f} m over the net")
        landing = facts.get("landing")
        if landing is not None and not (end is not None and number == end["shot"]):  # the end says how the last one came down
            if landing["pOverNet"] < 1.0 - CALL_CONFIDENCE:
                parts.append("did not get over the net")
            elif landing["pOverNet"] < CALL_CONFIDENCE:
                parts.append(f"whether it got over the net is too close to call ({_percent(landing['pOverNet'])})")
            else:
                parts.append(f"came down {landing['call']}" + (f" (in {_percent(landing['pIn'])})" if landing["call"] == TOO_CLOSE else ""))
        name = facts["name"]
        add("shot", hitter, [number], frames, name["call"], name["p"], facts, ", ".join(parts) + ".")

    for half, player in sorted((summary.get("players") or {}).items()):
        if player is None or not player.get("frames"):
            continue
        facts = {key: player[key] for key in ("standingReachM", "armReachM") if player.get(key) is not None}
        arm = player.get("racketArm")
        if arm is not None:
            facts["racketArm"] = arm
        theirs = [shot["shot"] for shot in shots if shot["hitter"] == half]
        if not facts or not theirs:
            continue
        reach = f"reaches {facts['armReachM']['value']:.2f} ± {facts['armReachM']['sigma']:.2f} m from the hips" if "armReachM" in facts else ""
        hand = f"plays {arm['call']}-handed ({_percent(arm['p'])})" if arm is not None and arm["call"] in ("left", "right") else ""
        said = " and ".join(part for part in (reach, hand) if part) or "was seen"
        add("player", half, theirs, [frame_of(n) for n in theirs], arm["call"] if arm is not None else TOO_CLOSE, arm["p"] if arm is not None else None, facts,
            f"The {half} player {said}.")
    return out


def _shot_facts(shot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """A shot's facts, each measurement with its standard deviation, and its name as said."""
    name = shot["name"]
    families = name["call"].split(" or ")
    facts: dict[str, Any] = {"name": {"call": name["call"], "p": round(sum(name["p"].get(f, 0.0) for f in families), 3), "families": name["p"]}}
    for detail in ("cross", "backhand", "jump"):
        if name.get(detail) is not None:
            facts[detail] = name[detail]
    for key in ("speedMS", "elevationDeg", "apexM"):
        if shot.get(key) is not None:
            facts[key] = shot[key]
    if shot.get("net") is not None:
        facts["netClearanceM"] = shot["net"]["clearanceM"]
    facts["contactHeightM"] = {"value": shot["contact"][2], "sigma": shot["contactSigma"][2]}
    if shot.get("landing") is not None:
        landing = shot["landing"]
        facts["landing"] = {"call": landing["call"], "pIn": landing["pIn"], "pOverNet": landing["pOverNet"], **_point(landing["point"], landing["covariance"])}
    if shot.get("pressure") is not None:
        facts["receiver"] = {"player": OTHER[shot["hitter"]], **shot["pressure"]}
    body = shot.get("body")
    if body is not None:
        facts["body"] = {"arm": body.get("arm"), **{key: body[key] for key in BODY if body.get(key) is not None}}
        ready = body.get("ready")
        if ready is not None:
            facts["body"]["ready"] = {key: ready[key] for key in READY if ready.get(key) is not None}
    if shot.get("players") is not None:
        facts["stood"] = {half: None if where is None else _point(where["point"], where["covariance"]) for half, where in shot["players"].items()}
    if shot.get("value") is not None:
        facts["chanceOfWinning"] = shot["value"]
    return facts, name["said"]


def _point(point: list[float], covariance: list[list[float]]) -> dict[str, dict[str, float]]:
    """A point on the floor as two measurements with their standard deviations."""
    return {
        "xM": {"value": round(point[0], 3), "sigma": round(math.sqrt(covariance[0][0]), 3)},
        "yM": {"value": round(point[1], 3), "sigma": round(math.sqrt(covariance[1][1]), 3)},
    }


def _percent(p: float) -> str:
    return f"{round(100 * p)}%"
