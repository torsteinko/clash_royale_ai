# M4 scenario report — gpusim (M4.1)

generated: 2026-09-15T22:11:17+00:00 · engine: gpusim@b8736bdbc1d2 · data: `/home/torromh_gmail_com/clash_royale_ai/fidelity/patched` · level 11 · dt 0.05s · 1500 ticks (75s)

**determinism: PASS — two full runs byte-identical**

| scenario | sha256 | first dmg →red | first dmg →blue | unit deaths | crowns (b-r) |
|---|---|---|---|---|---|
| duel_knight | a2e715ca42b5 | — | — | 1 | 0-0 |
| duel_musketeer | 028243c589c9 | t=17.30 princess_red_right | — | 2 | 0-0 |
| tower_press | 0ee2097b6302 | t=11.90 princess_red_left | — | 1 | 0-0 |
| spell_hit | f67e3172301c | t=7.10 princess_red_right | t=19.85 princess_blue_right | 1 | 0-0 |
| push_left | f2c81857aa23 | t=11.90 princess_red_left | — | 2 | 1-0 |

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
