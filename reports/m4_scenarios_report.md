# M4 scenario report — gpusim (M4.1)

generated: 2026-09-15T20:35:02+00:00 · engine: gpusim@aa3d909ec5e4 · data: `/home/torromh_gmail_com/clash_royale_ai/fidelity/patched` · level 11 · dt 0.05s · 1500 ticks (75s)

**determinism: PASS — two full runs byte-identical**

| scenario | sha256 | first dmg →red | first dmg →blue | unit deaths | crowns (b-r) |
|---|---|---|---|---|---|
| duel_knight | e65c47a28445 | — | — | 1 | 0-0 |
| duel_musketeer | b01f0d4e5aeb | t=17.10 princess_red_right | — | 2 | 0-0 |
| tower_press | c539c40d6fcf | t=12.10 princess_red_left | — | 1 | 0-0 |
| spell_hit | f0e05e3cafe5 | t=7.15 princess_red_right | t=20.55 princess_blue_right | 1 | 0-0 |
| push_left | 85cd229901f5 | t=12.10 princess_red_left | — | 2 | 1-0 |

Tower HP checkpoints (sum of living towers, seconds):

| scenario | blue t=0/15/30/45/60/75 | red t=0/15/30/45/60/75 |
|---|---|---|
| duel_knight | 10928 / 10928 / 10928 / 10928 / 10928 / 10928 | 10928 / 10928 / 10928 / 10928 / 10928 / 10928 |
| duel_musketeer | 10928 / 10928 / 10928 / 10928 / 10928 / 10928 | 10928 / 10928 / 9613 / 9613 / 9613 / 9613 |
| tower_press | 10928 / 10928 / 10928 / 10928 / 10928 / 10928 | 10928 / 10322 / 9514 / 9514 / 9514 / 9514 |
| spell_hit | 10928 / 10928 / 10524 / 10524 / 10524 / 10524 | 10928 / 10722 / 10722 / 10722 / 10722 / 10722 |
| push_left | 10928 / 10928 / 10928 / 10928 / 10928 / 10928 | 10928 / 10322 / 7087 / 6561 / 6561 / 6561 |

Raw numbers: `reports/m4_scenarios_report.json`. Per-tick traces: `fidelity/m4_traces/`.
Diff against the Java-side run: `python3 fidelity/m4_scenarios.py compare <java>.jsonl <gpusim>.jsonl`.
