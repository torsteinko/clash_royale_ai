# Tracked divergences from the Java reference sim

Rule: the GPU sim must not silently approximate. Every known behavioural
difference from `crforge` (the reference implementation) is listed here and
must either be fixed (by the milestone noted) or excluded from training
configs until then.

| # | Divergence | Status | Milestone |
|---|---|---|---|
| 1 | Movement is a straight march toward the enemy king tower; no bridge waypoints, no occupancy/collision, no pathing | OPEN | M2 |
| 2 | No combat yet (units cannot damage anything; towers passive) | OPEN | M2 |
| 3 | No targeting/aggro logic (sight range, retarget, building-only) | OPEN | M2 |
| 4 | Spells not implemented | OPEN | M3 |
| 5 | Deploy-time, placement-zone rules not enforced (deploy() trusts the caller) | OPEN | M3 |
| 6 | Overtime/tiebreak/win conditions not implemented (time just advances) | OPEN | M3 |
| 7 | Card cycle/hand not modelled (caller passes any card) | OPEN | M3 |
| 8 | Speed units: `speed*dt/20` is a placeholder mapping (crforge speed scale to be verified against the Java tick loop) | OPEN | M4 |
| 9 | Elixir: single-elixir phase only (no double/triple at 120s/240s) | OPEN | M3 |
| 10 | Unit spawn overflow (slots > MAX_UNITS) silently dropped | OPEN | M5 |
| 11 | `summonCharacterLevelIndex` not applied — Skeleton Army spawns base-level (32/32) skeletons instead of its level-index-5 variant (51/51). NOTE: the Java reference sim has the same gap (field parsed into cards.json but unused in its engine) — port the mechanic in both | OPEN | M3 |
| 12 | Minion Horde per-minion damage: noff shows 46 (+1 level factor) but game data has no level-shift for the horde — resolve via L2 probe before porting anything | OPEN | M3 |

Empty table = the sim is fully equivalent (M4 gate).
