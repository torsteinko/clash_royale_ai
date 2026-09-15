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
| 13 | King tower activation + attacks — RESOLVED (activates on damage/princess loss; 7.0 range, 109 dmg, 1.0s cd) | DONE | — |
| 14 | Projectiles: homing flight + radius impact DONE; remaining: non-homing/arc shots (gravity), AOE-on-impact, scatter/pierce/returning projectiles, tower shots still instant | PARTIAL | M2/M3 |
| 15 | Attack windup / loadTime — RESOLVED (AttackStateMachine port: windup = max(0, cd − load)) | DONE | — |
| 16 | Multiple targets / AOE splash / abilities (charge, dash, reflect, kamikaze, death spawns) not implemented | OPEN | M3 |
| 17 | Attack cadence: the Java reference accumulates windup in float32 and fires one tick late (25-tick cycles vs the intended 24). GPU sim uses an epsilon to land the intended cadence — diff ≤ 1 tick per attack; accepted within M4 timing tolerance | ACCEPTED | M4 |
| 18 | First-hit model (`first hit = hit time − load time`, load accrue during deploy/move): implemented from the reference's community-documented "secret stats" model. NOT yet verified against real-game timings — verify spawn→first-hit (e.g. knight 0.5s) via L2 probes/replays | PENDING-VERIFY | M4/L2 |

Resolved: #2 combat (melee/ranged, cooldowns, deaths) — DONE 2026-09-15;
#8 speed formula (`speed * 1000 / 60` game-units/s, from GameUnits.java) — VERIFIED 2026-09-15;
#9 elixir phases (double at 120s, triple at 240s) — DONE; #11/#12 data rulings applied;
#13 king activation + king attacks — DONE; #15 windup/loadTime (AttackStateMachine port) — DONE.

Empty table = the sim is fully equivalent (M4 gate).
