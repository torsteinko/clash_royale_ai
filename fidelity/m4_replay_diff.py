"""M4.2 — replay diff: recorded Java replays (record_replays.py format) replayed
in gpusim, diffed on tower HP + elixir.

Replay format (produced on the Java side by `~/record_replays.py`, files in
`~/replays/*.json` on clash-training): `meta` (deck names, seed, result, card
registry), `frames` (per 15-tick step: game time, both elixirs, entities
`[id, x, y, hp]`), `acts` (blue's plays as `[step, cardName, x, y]`).

Reference mechanics discovered while building this (see DIVERGENCES.md):
  * the Java hand shuffles the deck at reset (`Hand.java`: Collections.shuffle,
    blue = Random(seed), red = Random(seed+1)) — replicated bit-exactly in
    `fidelity/java_random.py`; the gpusim deck order is set to the same shuffle
    so both hands evolve identically (gpusim's hand+cycle model is an exact FIFO
    equivalent of the Java draw/queue model);
  * blue: every recorded action applied at its step at (x, y_flip); the slot is
    resolved against the modelled hand (shadow check) and the sim's hand;
  * red: reconstructed from *new-entity evidence*: entities first seen in frame
    t grouped by name; a group is accepted as a card play when the name maps to
    a card of red's recorded deck AND the group size equals that card's spawn
    count AND red's elixir budget (recorded elixir + regen model) can pay it.
    Everything else (spells, invisible in frames; ability spawns such as witch
    skeletons, which arrive 4 at a time vs the 3 of the skeletons card) is
    *flagged*, not replayed.

Coordinate note: the Java arena has blue at the top (blue crown tower y=3),
gpusim has blue at the bottom (y=29) — a pure y-reflection `32 - y`. x stays
as-is for both sides (left/right lane naming is x-ordered in both sims).

Usage:
  python3 fidelity/m4_replay_diff.py run <replay.json> [...] [--out reports/]
  python3 fidelity/m4_replay_diff.py run --dir ~/replays --limit 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpusim.cards import load_tables                       # noqa: E402
from gpusim.env import ELIXIR_PERIOD, ELIXIR_START, make_sim   # noqa: E402
from fidelity.java_random import shuffled_hand_order       # noqa: E402

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")
STEP_TICKS = 15
TICK_DT = 0.05
Y_FLIP = 32.0

# tower entity id -> trace slot (blue crown/princess L/R, red ...), from the
# Java entity registry: id 1 = blue crown (y=3), 2 = blue princess x<9, ...
TOWER_ID_SLOT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def norm_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


def load_replay(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    for k in ("meta", "frames", "acts"):
        assert k in d, f"{path}: missing '{k}'"
    assert d["meta"].get("blue_deck") and d["meta"].get("red_deck"), f"{path}: no deck names"
    return d


def _card_key(name: str, t: "CardTable") -> str | None:
    """Map a Java card/unit name to a gpusim card key (lowercase, no spaces)."""
    n = norm_name(name)
    if n in t.card_index:
        return n
    # unit name -> its card (e.g. 'minion' -> 'minions')
    for i, unit in enumerate(t.unit_names):
        if norm_name(unit) == n:
            for card, ui in enumerate(t.unit_of_card.tolist()):
                if ui == i and int(t.card_types[card]) in (0, 2):
                    return norm_name(t.names[card])
    for i, cname in enumerate(t.names):
        if norm_name(cname) == n:
            return norm_name(t.names[i])
    return None


def deck_indices(names: list[str], t: "CardTable", seed: int | None = None) -> list[int]:
    """Card indices for a deck; with `seed`, the Java reset shuffle is applied
    (blue order -> pass meta seed, red order -> pass seed+1)."""
    keys = []
    for nm in names:
        key = _card_key(nm, t)
        assert key is not None and key in t.card_index, f"unknown deck card {nm!r}"
        keys.append(key)
    assert len(keys) == 8, f"deck must be 8 cards: {names}"
    if seed is not None:
        keys = shuffled_hand_order(keys, int(seed))
    return [t.card_index[k] for k in keys]


def _deck_list(name: str) -> list[str]:
    """Deck name -> card list (see fidelity/decks.py)."""
    from fidelity.decks import DECKS

    assert name in DECKS, f"unknown deck name {name!r} (known: {sorted(DECKS)})"
    return DECKS[name]


def new_entities_by_step(frames: list[dict], card_by_id: dict) -> dict[int, list[dict]]:
    """step -> new (non-tower) entities first seen in that frame, incrementally."""
    seen: set[int] = set()
    out: dict[int, list[dict]] = {}
    for fr in frames:
        for e in fr["ents"]:
            eid = int(e[0])
            if eid in seen:
                continue
            seen.add(eid)
            c = card_by_id.get(eid)
            if c is None or c["type"] == "TOWER":
                continue
            out.setdefault(fr["t"], []).append({
                "id": eid, "name": c["name"], "team": c["team"],
                "key": None, "x": float(e[1]), "y": float(e[2]), "hp": float(e[3]),
            })
    return out


def replay_diff(path: str, t: "CardTable", tol_hp: float = 1.0, tol_elixir: float = 1.5,
                red_mode: str = "auto") -> dict:
    d = load_replay(path)
    meta, frames, acts = d["meta"], d["frames"], d["acts"]
    card_by_id = {int(k): v for k, v in meta["cards"].items()}
    seed = int(meta.get("seed") or 0)

    # Java reset shuffles: blue = Random(seed), red = Random(seed+1)
    deck_blue = deck_indices(_deck_list(meta["blue_deck"]), t, seed=seed)
    deck_red = deck_indices(_deck_list(meta["red_deck"]), t, seed=seed + 1)

    sim = make_sim(DATA, batch_size=1)
    sim.reset()
    sim.set_deck(0, deck_blue)
    sim.set_deck(1, deck_red)

    # shadow hand model: verify the recorded blue labels against shuffle + FIFO
    shadow = [t.names[i].lower().replace(" ", "") for i in deck_blue]
    shadow_hand, shadow_next, shadow_queue = shadow[:4], shadow[4], shadow[5:]
    labels_ok, labels_bad = 0, []

    ents_by_step = new_entities_by_step(frames, card_by_id)
    for lst in ents_by_step.values():
        for e in lst:
            e["key"] = _card_key(e["name"], t)

    acts_by_step: dict[int, list] = {}
    for a in acts:
        acts_by_step.setdefault(int(a[0]), []).append((a[1], float(a[2]), float(a[3])))

    red_deck_keys = {norm_name(t.names[i]) for i in deck_red}

    def regen_by_step(step: int) -> float:
        secs = step * STEP_TICKS * TICK_DT
        if secs <= 120.0:
            return secs / ELIXIR_PERIOD
        return 120.0 / ELIXIR_PERIOD + 2.0 * (secs - 120.0) / ELIXIR_PERIOD

    tracks = {"be": [], "re": [], "th": []}
    flags: list[dict] = []
    blue_applied = 0
    red_applied = 0
    red_skipped: list[dict] = []
    spend_recon = 0.0

    for fr in frames:
        step = fr["t"]
        if step == 0:
            continue

        # ---------------- blue: recorded action(s)
        for (name, ax, ay) in acts_by_step.get(step, []):
            key = _card_key(name, t)
            # shadow verification of the label against the shuffle+FIFO model
            skey = norm_name(name)
            if skey in shadow_hand:
                labels_ok += 1
                s = shadow_hand.index(skey)
                shadow_hand[s] = shadow_next
                shadow_queue.append(skey)
                shadow_next = shadow_queue.pop(0)
            else:
                labels_bad.append({"step": step, "label": name, "model_hand": list(shadow_hand)})

            ci = t.card_index.get(key) if key else None
            hand = sim.s.hand[0, 0].tolist()
            if ci is None or ci not in hand:
                # fallback: entity evidence (a blue unit of a hand card spawned this step)
                ents = {e["key"] for e in ents_by_step.get(step, []) if e["team"] == 0}
                cand = next((h for h in hand if t.names[h].lower().replace(" ", "") in ents), None)
                if cand is None:
                    flags.append({"step": step, "kind": "blue_label_unresolved", "detail": name})
                    continue
                flags.append({"step": step, "kind": "blue_label_from_entities", "detail": name})
                ci = cand
            slot = hand.index(ci)
            ok = sim.play(0, slot, torch.tensor([ax]), torch.tensor([Y_FLIP - ay]))
            if not bool(ok[0]):
                flags.append({"step": step, "kind": "blue_play_rejected",
                              "detail": f"{name} @({ax},{ay})"})
                continue
            blue_applied += 1

        # ---------------- red: reconstructed spawn-evidence plays
        if red_mode == "auto":
            groups: dict[str, list[dict]] = {}
            for cand in ents_by_step.get(step, []):
                if cand["team"] == 1:
                    groups.setdefault(cand["key"] or cand["name"], []).append(cand)
            for key, cands in groups.items():
                if key is None or key not in red_deck_keys:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": "not-in-red-deck"})
                    continue
                ci = t.card_index[key]
                count = int(t.spawn_count[ci])
                if count > 1 and len(cands) != count:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": f"spawn-count {len(cands)} != {count} (ability spawn?)"})
                    continue
                if count == 1 and len(cands) > 1:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": f"{len(cands)} simultaneous spawns of a 1-count card"})
                    continue
                cost = float(t.costs[ci])
                spend_obs = max(0.0, ELIXIR_START + regen_by_step(step) - fr["re"])
                if spend_recon + cost > spend_obs + 2.0:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": "elixir-budget (unexplained red spend)"})
                    continue
                offs = t.formation_offsets[ci]
                dx, dy = offs[0] if offs else (0.0, 0.0)
                px, py = cands[0]["x"] - dx, Y_FLIP - (cands[0]["y"] - dy)
                hand_red = sim.s.hand[0, 1].tolist()
                if ci not in hand_red:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": "not-in-red-hand"})
                    continue
                ok = sim.play(1, hand_red.index(ci), torch.tensor([px]), torch.tensor([py]))
                if bool(ok[0]):
                    red_applied += 1
                    spend_recon += cost
                else:
                    red_skipped.append({"step": step, "name": cands[0]["name"],
                                        "reason": "red_play_rejected"})

        # ---------------- tick + compare
        sim.tick(STEP_TICKS)
        s = sim.s
        rec_th = {TOWER_ID_SLOT[int(e[0])]: float(e[3]) for e in fr["ents"]
                  if int(e[0]) in TOWER_ID_SLOT}
        tracks["be"].append({"step": step, "sim": float(s.elixir[0, 0]), "rec": fr["be"]})
        tracks["re"].append({"step": step, "sim": float(s.elixir[0, 1]), "rec": fr["re"]})
        tracks["th"].append({"step": step, "sim": s.tower_hp[0].tolist(), "rec": rec_th})

    # ---------------- summarise
    def track_summary_th(rows) -> dict:
        mx, first, over = 0.0, None, 0
        for r in rows:
            if not r["rec"]:
                continue
            dd = max(abs(r["sim"][slot] - hp) for slot, hp in r["rec"].items())
            mx = max(mx, dd)
            if dd > tol_hp:
                over += 1
                if first is None:
                    first = r["step"]
        return {"max_abs_diff": round(mx, 3), "first_divergent_step": first,
                "steps_over_tol": over}

    def track_summary_el(rows) -> dict:
        mx, first, over = 0.0, None, 0
        for r in rows:
            dd = abs(r["sim"] - r["rec"])
            mx = max(mx, dd)
            if dd > tol_elixir:
                over += 1
                if first is None:
                    first = r["step"]
        return {"max_abs_diff": round(mx, 3), "first_divergent_step": first,
                "steps_over_tol": over}

    s = sim.s
    summary = {
        "format": "m4-replay-diff-v1",
        "file": os.path.basename(path),
        "blue_deck": meta["blue_deck"], "red_deck": meta["red_deck"],
        "seed": seed, "recorded_result": meta.get("result"),
        "steps": len(frames) - 1,
        "blue_actions_applied": blue_applied,
        "red_actions_applied": red_applied,
        "blue_labels_total": labels_ok + len(labels_bad),
        "blue_labels_model_match": labels_ok,
        "blue_labels_model_fail": labels_bad[:5],
        "flags": flags, "red_skipped": red_skipped,
        "tower_hp": track_summary_th(tracks["th"]),
        "blue_elixir": track_summary_el(tracks["be"]),
        "red_elixir": track_summary_el(tracks["re"]),
        "final": {"towers_alive": [bool(x) for x in s.tower_alive[0].tolist()],
                  "crowns": s.crowns[0].tolist(), "winner": int(s.winner[0]),
                  "units_alive": int(s.u_active[0].sum())},
        "tol": {"tower_hp": tol_hp, "elixir": tol_elixir},
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_run = sub.add_parser("run", help="replay + diff one or more replay files")
    ap_run.add_argument("files", nargs="*")
    ap_run.add_argument("--dir", default="", help="directory of replay files")
    ap_run.add_argument("--limit", type=int, default=0)
    ap_run.add_argument("--glob", default="*.json")
    ap_run.add_argument("--out", default="reports")
    ap_run.add_argument("--tol-hp", type=float, default=1.0)
    ap_run.add_argument("--tol-elixir", type=float, default=1.5)
    ap_run.add_argument("--red-mode", default="auto", choices=["auto", "none"])
    args = ap.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(os.path.expanduser(args.dir), args.glob)))
    if args.limit:
        files = files[: args.limit]
    assert files, "no replay files given"

    t = load_tables(DATA)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for f in files:
        t0 = time.perf_counter()
        summ = replay_diff(f, t, tol_hp=args.tol_hp, tol_elixir=args.tol_elixir,
                           red_mode=args.red_mode)
        summ["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(summ)
        print(f"{summ['file']}: steps {summ['steps']}, blue acts {summ['blue_actions_applied']}/"
              f"{summ['blue_labels_total']} (model-match {summ['blue_labels_model_match']}), "
              f"red acts {summ['red_actions_applied']}, flags {len(summ['flags'])}, "
              f"skipped-red {len(summ['red_skipped'])} | tower_hp max {summ['tower_hp']['max_abs_diff']} "
              f"(first>{summ['tower_hp']['first_divergent_step']}) | be max {summ['blue_elixir']['max_abs_diff']} "
              f"re max {summ['red_elixir']['max_abs_diff']} | {summ['seconds']}s")

    out_json = os.path.join(args.out, "m4_replay_diff.json")
    with open(out_json, "w") as f:
        json.dump({"format": "m4-replay-diff-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "data_dir": DATA, "tolerances": {"tower_hp": args.tol_hp, "elixir": args.tol_elixir},
                   "replays": results}, f, indent=1)
    md = os.path.join(args.out, "m4_replay_diff.md")
    with open(md, "w") as f:
        f.write("# M4.2 replay diff — recorded Java replays replayed in gpusim\n\n")
        f.write(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} · data dir `{DATA}` · "
                f"tolerances: tower HP ±{args.tol_hp}, elixir ±{args.tol_elixir}\n\n")
        f.write("| replay | steps | blue acts | labels model-match | red acts | red skipped | "
                "tower-HP max diff | first div. | be max | re max |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['file']} | {r['steps']} | {r['blue_actions_applied']} | "
                    f"{r['blue_labels_model_match']}/{r['blue_labels_total']} | "
                    f"{r['red_actions_applied']} | {len(r['red_skipped'])} | "
                    f"{r['tower_hp']['max_abs_diff']} | {r['tower_hp']['first_divergent_step']} | "
                    f"{r['blue_elixir']['max_abs_diff']} | {r['red_elixir']['max_abs_diff']} |\n")
        for r in results:
            f.write(f"\n## {r['file']}\n\n")
            f.write(f"- blue deck `{r['blue_deck']}` vs red deck `{r['red_deck']}` (seed {r['seed']}), "
                    f"recorded result **{r['recorded_result']}**, {r['steps']} steps, replayed in {r['seconds']}s\n")
            f.write(f"- recorded blue labels explained by the shuffle+FIFO hand model: "
                    f"{r['blue_labels_model_match']}/{r['blue_labels_total']}"
                    + (f" — failures: {json.dumps(r['blue_labels_model_fail'][:3])}"
                       if r['blue_labels_model_fail'] else "") + "\n")
            f.write(f"- tower-HP diff: max {r['tower_hp']['max_abs_diff']}, first over tol at step "
                    f"{r['tower_hp']['first_divergent_step']}, steps over tol {r['tower_hp']['steps_over_tol']}\n")
            f.write(f"- elixir diff: blue max {r['blue_elixir']['max_abs_diff']}, "
                    f"red max {r['red_elixir']['max_abs_diff']}\n")
            f.write(f"- final sim state: winner {r['final']['winner']}, crowns {r['final']['crowns']}, "
                    f"towers alive {r['final']['towers_alive']}, units alive {r['final']['units_alive']}\n")
            if r["flags"]:
                f.write(f"- blue-side flags: {json.dumps(r['flags'][:8])}\n")
            if r["red_skipped"]:
                kinds: dict[str, int] = {}
                for sk in r["red_skipped"]:
                    kinds[sk["reason"]] = kinds.get(sk["reason"], 0) + 1
                f.write(f"- red plays not replayed (unobservable/unmodelled): {kinds}\n")
    print(f"\nwrote {out_json}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
