# Tracked divergences from the Java reference sim

Rule: the GPU sim must not silently approximate. Every known behavioural
difference from `crforge` (the reference implementation) is listed here and
must either be fixed (by the milestone noted) or excluded from training
configs until then.

| # | Divergence | Status | Milestone |
|---|---|---|---|
| 1 | Movement: target-chase / enemy-king advance DONE; bridge waypoint routing DONE (M2, test_pathing). Remaining: unit collision/occupancy resolution, jump-river speed | PARTIAL | M2/M3 |
| 3 | No collision/occupancy between units (units overlap freely) | OPEN | M2 |
| 4 | Spells: damage + radius + crown-% core DONE (M3.3, verified); remaining: projectile flight time (M3.4), ticking zones poison/earthquake (M3.5), log model (M3.6) | PARTIAL | M3 |
| 5 | Placement zones + deploy timers enforced via `play()` (M3.2, test_m3); direct `deploy()` API still trusts the caller (tests/scripts only — the policy path is play()) | PARTIAL | M3/M4 |
| 6 | Overtime end uses crown/tower-HP rules (simplified); no per-tick edge cases | PARTIAL | M3 |
| 7 | Card cycle/hand modelled (M3.1: play() rotates played card to the back of the 8-card cycle, test_m3) — DONE; deck-order = hand[0:4] + cycle[4:8] per the reference | DONE | — |
| 10 | Unit spawn overflow (slots > MAX_UNITS) silently dropped | OPEN | M5 |
| 13 | King tower activation + attacks — RESOLVED (activates on damage/princess loss; 7.0 range, 109 dmg, 1.0s cd) | DONE | — |
| 14 | Projectiles: homing flight + radius impact DONE; remaining: non-homing/arc shots (gravity), AOE-on-impact, scatter/pierce/returning projectiles, tower shots still instant | PARTIAL | M2/M3 |
| 15 | Attack windup / loadTime — RESOLVED (AttackStateMachine port: windup = max(0, cd − load)) | DONE | — |
| 16 | Multiple targets / AOE splash / abilities (charge, dash, reflect, kamikaze, death spawns) not implemented | OPEN | M3 |
| 17 | Attack cadence: the Java reference accumulates windup in float32 and fires one tick late (25-tick cycles vs the intended 24). GPU sim uses an epsilon to land the intended cadence — diff ≤ 1 tick per attack; accepted within M4 timing tolerance | ACCEPTED | M4 |
| 18 | First-hit model (`first hit = hit time − load time`, load accrue during deploy/move): implemented from the reference's community-documented "secret stats" model. NOT yet verified against real-game timings — verify spawn→first-hit (e.g. knight 0.5s) via L2 probes/replays | PENDING-VERIFY | M4/L2 |
| 19 | Same-tick resolution: all attacks of a tick apply, deaths resolve at end of tick — mirror duel (`duel_knight`) ends in a mutual KO on the identical tick. Java source shows the same shape (`gameState.processDeaths()` after combat, no mid-tick alive re-check) — confirm via the M4.2 trace diff (canary: both knights must die at tick 301 in both sims) | PENDING-VERIFY | M4 |
| 20 | Overkill: damage could push HP below 0 (tower ran to −68 on the killing blow) — FIXED in M4.1: damage clamps at 0, matching Java `Health.takeDamage` (`current -= min(damage, current)`); covered by test_m4 (`push_left` tower ends at exactly 0.0) | DONE | — |

Resolved: #2 combat (melee/ranged, cooldowns, deaths) — DONE 2026-09-15;
#8 speed formula (`speed * 1000 / 60` game-units/s, from GameUnits.java) — VERIFIED 2026-09-15;
#9 elixir phases (double at 120s, triple at 240s) — DONE; #11/#12 data rulings applied;
#13 king activation + king attacks — DONE; #15 windup/loadTime (AttackStateMachine port) — DONE.

Empty table = the sim is fully equivalent (M4 gate).
