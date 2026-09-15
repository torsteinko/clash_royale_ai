# Tracked divergences from the Java reference sim

Rule: the GPU sim must not silently approximate. Every known behavioural
difference from `crforge` (the reference implementation) is listed here and
must either be fixed (by the milestone noted) or excluded from training
configs until then.

| # | Divergence | Status | Milestone |
|---|---|---|---|
| 1 | Movement: straight march (target chase / enemy-king advance); no bridge waypoints, no occupancy, no collision resolution, no jump-river speed | OPEN | M2 |
| 3 | No collision/occupancy between units (units overlap freely) | OPEN | M2 |
| 4 | Spells not implemented | OPEN | M3 |
| 5 | Deploy-time, placement-zone rules not enforced (deploy() trusts the caller) | OPEN | M3 |
| 6 | Overtime end uses crown/tower-HP rules (simplified); no per-tick edge cases | PARTIAL | M3 |
| 7 | Card cycle/hand not modelled (caller passes any card) | OPEN | M3 |
| 10 | Unit spawn overflow (slots > MAX_UNITS) silently dropped | OPEN | M5 |
| 13 | King tower: never attacks; activation rule (damage taken / princess falls) not implemented | OPEN | M2 |
| 14 | Projectiles: homing flight + radius impact DONE; remaining: non-homing/arc shots (gravity), AOE-on-impact, scatter/pierce/returning projectiles, tower shots still instant | PARTIAL | M2/M3 |
| 15 | Attack windup / loadTime not modelled (attack fires the instant the cooldown expires) | OPEN | M2 |
| 16 | Multiple targets / AOE splash / abilities (charge, dash, reflect, kamikaze, death spawns) not implemented | OPEN | M3 |

Resolved: #2 combat (melee/ranged, cooldowns, deaths) — DONE 2026-09-15;
#8 speed formula (`speed * 1000 / 60` game-units/s, from GameUnits.java) — VERIFIED 2026-09-15;
#9 elixir phases (double at 120s, triple at 240s) — DONE; #11/#12 data rulings applied.

Empty table = the sim is fully equivalent (M4 gate).
