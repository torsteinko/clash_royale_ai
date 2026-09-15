#!/usr/bin/env python3
"""M4.1 — fixed-scenario runner: scripted battles -> machine-readable traces.

Runs a fixed set of deterministic scenarios (duels, tower pressure, spell hit)
in one batched `BatchedCRSim` and writes, per scenario:

    fidelity/m4_traces/<name>.scenario.json   manifest (decks + action schedule)
    fidelity/m4_traces/<name>.gpusim.jsonl    per-tick trace (m4-trace-v1)

plus an aggregate report (`reports/m4_scenarios_report.{md,json}`).

The manifest + trace formats are the differential-harness contract: the same
scenarios can be replayed against the Java reference (crforge gym-bridge with
`ticks_per_step=1`) and the two traces diffed with `compare`. Full spec:
docs/M4_TRACE_FORMAT.md.

Determinism: the sim core is pure (no RNG); `run` executes the scenario batch
twice and flags/reports unless both passes are byte-identical.

CLI:
    python3 fidelity/m4_scenarios.py run [--out DIR] [--report-dir DIR]
    python3 fidelity/m4_scenarios.py compare A.jsonl B.jsonl [--tol T]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpusim.env import TICK_DT, make_sim  # noqa: E402

TRACE_FORMAT = "m4-trace-v1"
MANIFEST_FORMAT = "m4-scenario-v1"
REPORT_FORMAT = "m4-report-v1"

DURATION_TICKS = 1500          # 75 s at 20 tps
CHECKPOINT_TICKS = (0, 300, 600, 900, 1200, 1500)   # every 15 s

# One shared deck for every scenario and both sides. The first 4 cards are the
# opening hand, so scenario actions use slots 0..2 only — no rotation drift.
DECK = ["Knight", "Musketeer", "Fireball", "Archer", "Giant", "Zap", "Log", "Cannon"]

TOWER_SLOTS = [
    "king_blue", "princess_blue_left", "princess_blue_right",
    "king_red", "princess_red_left", "princess_red_right",
]


@dataclass(frozen=True)
class Action:
    """One scripted play. `at_tick=k` = applied between record k and k+1."""

    at_tick: int
    side: int          # 0 blue, 1 red
    slot: int          # hand slot 0..3
    x: float
    y: float
    card: str          # expected card name (asserted while running)

    def to_json(self) -> dict:
        return {
            "at_tick": self.at_tick,
            "side": self.side,
            "side_name": "blue" if self.side == 0 else "red",
            "slot": self.slot,
            "x": self.x,
            "y": self.y,
            "card": self.card,
        }


@dataclass(frozen=True)
class Scenario:
    name: str
    desc: str
    actions: tuple[Action, ...]
    deck: tuple[str, ...] | None = None  # None = the shared DECK


ZONE_DECK = ("Knight", "Poison", "Earthquake", "Minions", "Fireball", "Giant",
             "Musketeer", "Zap")  # ordered so both the gpusim hand (deck[0:4]) and the
             # java-shuffled hands (java_random, seeds 7/8) hold every played card


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "duel_knight",
        "mirror melee duel on the right bridge lane: identical cadence, mutual KO on the same "
        "tick (t=15.05) — canary for same-tick damage/death resolution vs the Java reference",
        (Action(20, 0, 0, 14.5, 20.0, "Knight"),
         Action(20, 1, 0, 14.5, 12.0, "Knight")),
    ),
    Scenario(
        "duel_musketeer",
        "ranged vs melee duel (musketeer vs knight): projectile flight kills the knight; the "
        "musketeer then lands one hit on the red right princess tower before tower fire kills it",
        (Action(20, 0, 1, 14.5, 21.0, "Musketeer"),
         Action(20, 1, 0, 14.5, 12.0, "Knight")),
    ),
    Scenario(
        "tower_press",
        "single knight marches the left lane into the red left princess tower: first melee hit at "
        "t=11.05, tower fire kills the knight at t=16.85 (5 hits dealt)",
        (Action(20, 0, 0, 4.0, 17.5, "Knight"),),
    ),
    Scenario(
        "spell_hit",
        "fireball on a red knight standing at the red right princess tower: full damage to the "
        "unit, crown-tower % (206) to the tower; the knight then marches a full lane "
        "(deploy at (14.5, 8.5): (14.5, 8.0) is a Java TOWER tile and is rejected by "
        "Arena.isValidPlacement)",
        (Action(60, 1, 0, 14.5, 8.5, "Knight"),
         Action(80, 0, 2, 14.5, 8.5, "Fireball")),
    ),
    Scenario(
        "push_left",
        "knight + musketeer push the left lane: the red left princess tower falls (crown 1-0) "
        "before both attackers die to tower fire",
        (Action(20, 0, 0, 4.0, 17.5, "Knight"),
         Action(140, 0, 1, 4.0, 20.0, "Musketeer")),
    ),
    Scenario(
        "poison_zone",
        "M3.5 ticking zone vs the Java reference: blue poisons the red left princess lane; the "
        "red knight and minions walk through the zone (poison hits ground AND air)",
        (Action(20, 1, 0, 4.0, 8.5, "Knight"),
         Action(80, 1, 3, 5.0, 8.5, "Minions"),
         Action(60, 0, 1, 3.5, 7.0, "Poison")),
        deck=ZONE_DECK,
    ),
    Scenario(
        "earthquake_zone",
        "M3.5 ticking zone (hitsAir false): the red knight takes earthquake ticks, the minions "
        "fly through unharmed; the princess tower takes the building bonus (x4.5) + crown %",
        (Action(20, 1, 0, 4.0, 8.5, "Knight"),
         Action(80, 1, 3, 5.0, 8.5, "Minions"),
         Action(60, 0, 2, 3.5, 7.0, "Earthquake")),
        deck=ZONE_DECK,
    ),
)

_HIST_KEYS = ("time", "elixir", "tower_hp", "tower_alive", "crowns", "game_over",
              "winner", "u_active", "u_side", "u_card", "u_x", "u_y", "u_hp")


def _snapshot(s) -> dict:
    return {k: getattr(s, k).clone() for k in _HIST_KEYS}


def _r(v, n: int) -> float:
    return round(float(v), n)


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- runner
def run_scenarios(data_dir: str | Path | None = None,
                  duration_ticks: int = DURATION_TICKS,
                  level: int = 11,
                  device: str = "cpu") -> dict:
    """Run all SCENARIOS in one batched sim. Returns {name: {"records": [...], "text": str}}."""
    data_dir = Path(data_dir) if data_dir is not None else REPO_ROOT / "fidelity" / "patched"
    n_env = len(SCENARIOS)
    sim = make_sim(str(data_dir), batch_size=n_env, device=device, level=level)

    deck_idx = [sim.t.card_index[name.lower().replace(" ", "")] for name in DECK]
    for side in (0, 1):
        sim.set_deck(side, deck_idx)
    # per-scenario deck overrides (zone scenarios need Poison/Earthquake in hand)
    for ei, sc in enumerate(SCENARIOS):
        if sc.deck is None:
            continue
        mask = torch.zeros(n_env, dtype=torch.bool)
        mask[ei] = True
        idx = [sim.t.card_index[name.lower().replace(" ", "")] for name in sc.deck]
        for side in (0, 1):
            sim.set_deck(side, idx, env_mask=mask)

    schedule: dict[int, list[tuple[int, Action]]] = defaultdict(list)
    for ei, sc in enumerate(SCENARIOS):
        for a in sc.actions:
            schedule[a.at_tick].append((ei, a))

    hist = [_snapshot(sim.s)]
    for k in range(duration_ticks):
        for ei, a in schedule.get(k, ()):
            hand_card = int(sim.s.hand[ei, a.side, a.slot])
            got = sim.t.names[hand_card] if hand_card >= 0 else "<empty>"
            if got.lower() != a.card.lower():
                raise AssertionError(
                    f"{SCENARIOS[ei].name}: action at tick {k} target slot {a.slot} "
                    f"expected {a.card}, hand has {got}")
            mask = torch.zeros(n_env, dtype=torch.bool)
            mask[ei] = True
            x = torch.zeros(n_env)
            x[ei] = a.x
            y = torch.zeros(n_env)
            y[ei] = a.y
            ok = sim.play(a.side, a.slot, x, y, env_mask=mask)
            if not bool(ok[ei]):
                raise AssertionError(
                    f"{SCENARIOS[ei].name}: action at tick {k} rejected "
                    f"(elixir/zone) — elixir={float(sim.s.elixir[ei, a.side]):.2f}")
        sim.tick(1)
        hist.append(_snapshot(sim.s))

    runs = {}
    for ei, sc in enumerate(SCENARIOS):
        records = _serialize_env(hist, ei, sim.t.names)
        runs[sc.name] = {"records": records, "text": _trace_text(records)}
    return runs


def _trace_text(records: list[dict]) -> str:
    return "".join(json.dumps(r, separators=(",", ":"), sort_keys=False) + "\n" for r in records)


def _serialize_env(hist: list[dict], e: int, names: list[str]) -> list[dict]:
    out = []
    for i, snap in enumerate(hist):
        rec = {
            "i": i,
            # trace time from the tick index (Java runner: round(i * 0.05, 3));
            # the fp32 sim clock accumulates a ~0.001 s drift that used to leak
            # into the trace t field only.
            "t": _r(i * TICK_DT, 3),
            "el": [_r(snap["elixir"][e, 0], 3), _r(snap["elixir"][e, 1], 3)],
            "th": [_r(v, 2) for v in snap["tower_hp"][e].tolist()],
            "ta": [int(v) for v in snap["tower_alive"][e].tolist()],
            "cr": [int(snap["crowns"][e, 0]), int(snap["crowns"][e, 1])],
            "go": int(snap["game_over"][e]),
            "w": int(snap["winner"][e]),
        }
        slots = snap["u_active"][e].nonzero(as_tuple=False).flatten().tolist()
        if slots:
            units = [{
                "s": int(snap["u_side"][e, sl]),
                "c": names[int(snap["u_card"][e, sl])].lower(),
                "x": _r(snap["u_x"][e, sl], 3),
                "y": _r(snap["u_y"][e, sl], 3),
                "hp": _r(snap["u_hp"][e, sl], 2),
            } for sl in slots]
            units.sort(key=lambda u: (u["s"], u["x"], u["y"]))
            rec["u"] = units
        out.append(rec)
    return out


# ------------------------------------------------------------------- manifest
def build_manifest(scenario: Scenario, duration_ticks: int, level: int,
                   data_dir: str | Path, device: str = "cpu") -> dict:
    return {
        "format": MANIFEST_FORMAT,
        "name": scenario.name,
        "desc": scenario.desc,
        "source": "gpusim",
        "level": level,
        "dt": TICK_DT,
        "duration_ticks": duration_ticks,
        "trace_format": TRACE_FORMAT,
        "trace_file": f"{scenario.name}.gpusim.jsonl",
        "tick_semantics": (
            "record i = state after i ticks (t = i*dt, t=0 before any tick); an action with "
            "at_tick=k is applied between record k and record k+1, i.e. during tick k+1 "
            "(= Java side: submit it in step call number k+1, ticks_per_step=1)"
        ),
        "tower_slots": TOWER_SLOTS,
        "decks": {"blue": list(scenario.deck or DECK), "red": list(scenario.deck or DECK)},
        "actions": [a.to_json() for a in scenario.actions],
        "engine": {"impl": "gpusim", "commit": _git_sha(), "device": device},
        "data_dir": str(data_dir),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------- metrics / report
def scenario_metrics(records: list[dict], sha256: str) -> dict:
    th0 = records[0]["th"]
    blue_sl, red_sl = (0, 1, 2), (3, 4, 5)

    def sum_alive(rec, slots):
        return _r(sum(rec["th"][i] for i in slots if rec["ta"][i]), 2)

    checkpoints = [{
        "tick": k, "t": records[k]["t"],
        "tower_hp_blue": sum_alive(records[k], blue_sl),
        "tower_hp_red": sum_alive(records[k], red_sl),
    } for k in CHECKPOINT_TICKS if k < len(records)]

    first_damage = {}
    for side, slots in (("blue", blue_sl), ("red", red_sl)):
        hit = None
        for rec in records:
            for i in slots:
                if rec["th"][i] < th0[i] - 1e-6:
                    hit = {"tick": rec["i"], "t": rec["t"],
                           "tower": TOWER_SLOTS[i], "hp": rec["th"][i]}
                    break
            if hit:
                break
        first_damage[side] = hit

    n_units = [len(r.get("u", [])) for r in records]
    unit_deaths = sum(1 for a, b in zip(n_units, n_units[1:]) if b < a)
    last = records[-1]
    return {
        "ticks": len(records),
        "sha256": sha256,
        "checkpoints": checkpoints,
        "first_damage": first_damage,
        "unit_deaths": unit_deaths,
        "final": {
            "tower_hp": last["th"],
            "towers_alive": last["ta"],
            "crowns": last["cr"],
            "winner": last["w"],
            "units_alive": n_units[-1],
        },
    }


def build_report(runs: dict, shas: dict, deterministic: bool,
                 duration_ticks: int, level: int, data_dir: str | Path) -> dict:
    return {
        "format": REPORT_FORMAT,
        "source": "gpusim",
        "engine": {"impl": "gpusim", "commit": _git_sha()},
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "level": level,
        "dt": TICK_DT,
        "duration_ticks": duration_ticks,
        "deterministic": deterministic,
        "scenarios": [
            {"name": sc.name, "desc": sc.desc,
             **scenario_metrics(runs[sc.name]["records"], shas[sc.name])}
            for sc in SCENARIOS
        ],
    }


def render_report_md(rep: dict) -> str:
    dur_s = rep["duration_ticks"] * rep["dt"]
    L = []
    L.append("# M4 scenario report — gpusim (M4.1)")
    L.append("")
    L.append(f"generated: {rep['generated']} · engine: gpusim@{rep['engine']['commit']} · "
             f"data: `{rep['data_dir']}` · level {rep['level']} · dt {rep['dt']}s · "
             f"{rep['duration_ticks']} ticks ({dur_s:.0f}s)")
    L.append("")
    L.append(f"**determinism: {'PASS — two full runs byte-identical' if rep['deterministic'] else 'FAIL'}**")
    L.append("")
    L.append("| scenario | sha256 | first dmg →red | first dmg →blue | unit deaths | crowns (b-r) |")
    L.append("|---|---|---|---|---|---|")
    for sc in rep["scenarios"]:
        fd_b = sc["first_damage"]["red"]
        fd_r = sc["first_damage"]["blue"]
        fds = lambda h: f"t={h['t']:.2f} {h['tower']}" if h else "—"   # noqa: E731
        L.append(f"| {sc['name']} | {sc['sha256'][:12]} | {fds(fd_b)} | {fds(fd_r)} | "
                 f"{sc['unit_deaths']} | {sc['final']['crowns'][0]}-{sc['final']['crowns'][1]} |")
    L.append("")
    L.append("Tower HP checkpoints (sum of living towers, seconds):")
    L.append("")
    L.append("| scenario | blue t=0/15/30/45/60/75 | red t=0/15/30/45/60/75 |")
    L.append("|---|---|---|")
    for sc in rep["scenarios"]:
        def row(side):
            vals = " / ".join(f"{c['tower_hp_' + side]:.0f}" for c in sc["checkpoints"])
            return vals
        L.append(f"| {sc['name']} | {row('blue')} | {row('red')} |")
    L.append("")
    L.append("Raw numbers: `reports/m4_scenarios_report.json`. Per-tick traces: `fidelity/m4_traces/`.")
    L.append("Diff against the Java-side run: `python3 fidelity/m4_scenarios.py compare <java>.jsonl <gpusim>.jsonl`.")
    L.append("")
    return "\n".join(L)


# -------------------------------------------------------------------- outputs
def write_outputs(runs: dict, out_dir: Path, report_dir: Path,
                  duration_ticks: int, level: int, data_dir: Path,
                  deterministic: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    shas = {}
    for sc in SCENARIOS:
        text = runs[sc.name]["text"]
        shas[sc.name] = hashlib.sha256(text.encode()).hexdigest()
        (out_dir / f"{sc.name}.gpusim.jsonl").write_text(text)
        manifest = build_manifest(sc, duration_ticks, level, data_dir)
        (out_dir / f"{sc.name}.scenario.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rep = build_report(runs, shas, deterministic, duration_ticks, level, data_dir)
    (report_dir / "m4_scenarios_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (report_dir / "m4_scenarios_report.md").write_text(render_report_md(rep))
    return {"report": rep, "sha256": shas}


# --------------------------------------------------------------------- compare
def _load_trace(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_traces(path_a: str | Path, path_b: str | Path, tol: float = 0.0) -> dict:
    ra, rb = _load_trace(path_a), _load_trace(path_b)
    n = min(len(ra), len(rb))
    keys = ("t", "el", "th", "cr", "u_count", "u_x", "u_y", "u_hp")
    maxd = {k: 0.0 for k in keys}
    first = None
    for i in range(n):
        a, b = ra[i], rb[i]
        dd = {
            "t": abs(a["t"] - b["t"]),
            "el": max(abs(p - q) for p, q in zip(a["el"], b["el"])),
            "th": max(abs(p - q) for p, q in zip(a["th"], b["th"])),
            "cr": float(max(abs(p - q) for p, q in zip(a["cr"], b["cr"]))),
            "u_count": float(abs(len(a.get("u", [])) - len(b.get("u", [])))),
            "u_x": 0.0, "u_y": 0.0, "u_hp": 0.0,
        }
        ua, ub = a.get("u", []), b.get("u", [])
        if len(ua) == len(ub):
            for u1, u2 in zip(ua, ub):
                dd["u_x"] = max(dd["u_x"], abs(u1["x"] - u2["x"]))
                dd["u_y"] = max(dd["u_y"], abs(u1["y"] - u2["y"]))
                dd["u_hp"] = max(dd["u_hp"], abs(u1["hp"] - u2["hp"]))
        for k, v in dd.items():
            maxd[k] = max(maxd[k], v)
        if first is None and any(v > tol for v in dd.values()):
            first = i
    return {
        "ticks": [len(ra), len(rb)],
        "first_divergence_tick": first,
        "max_abs_diff": {k: round(v, 6) for k, v in maxd.items()},
    }


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4.1 fixed-scenario runner (gpusim side)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run scenarios twice, write traces + report")
    p_run.add_argument("--out", default=str(REPO_ROOT / "fidelity" / "m4_traces"))
    p_run.add_argument("--report-dir", default=str(REPO_ROOT / "reports"))
    p_run.add_argument("--duration", type=int, default=DURATION_TICKS)
    p_run.add_argument("--device", default="cpu")

    p_cmp = sub.add_parser("compare", help="diff two m4-trace-v1 jsonl files")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.add_argument("--tol", type=float, default=0.0)

    args = ap.parse_args(argv)

    if args.cmd == "compare":
        res = compare_traces(args.a, args.b, tol=args.tol)
        print(json.dumps(res, indent=2))
        ok = res["first_divergence_tick"] is None
        return 0 if ok else 1

    out_dir = Path(args.out)
    report_dir = Path(args.report_dir)
    data_dir = REPO_ROOT / "fidelity" / "patched"
    print(f"running {len(SCENARIOS)} scenarios ({args.duration} ticks = {args.duration * TICK_DT:.0f}s each)...")
    first = run_scenarios(data_dir=data_dir, duration_ticks=args.duration, device=args.device)
    print("determinism pass 2/2...")
    second = run_scenarios(data_dir=data_dir, duration_ticks=args.duration, device=args.device)
    deterministic = all(first[k]["text"] == second[k]["text"] for k in first)
    if not deterministic:
        print("!! determinism FAILED (runs differ) — writing evidence anyway")
    info = write_outputs(first, out_dir, report_dir, args.duration, 11, data_dir, deterministic)
    for name, sha in info["sha256"].items():
        print(f"  {name}: {sha[:16]}")
    print(f"traces -> {out_dir}\nreport -> {report_dir / 'm4_scenarios_report.md'}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
