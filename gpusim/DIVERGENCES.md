# Tracked divergences from the Java reference sim

Rule: the GPU sim must not silently approximate. Every known behavioural
difference from `crforge` (the reference implementation) is listed here and
must either be fixed (by the milestone noted) or excluded from training
configs until then.

| # | Divergence | Status | Milestone |
|---|---|---|---|
| 1 | Movement: target-chase / enemy-king advance DONE; bridge waypoint routing DONE (M2, test_pathing). Remaining: unit collision/occupancy resolution, jump-river speed | PARTIAL | M2/M3 |
| 3 | No collision/occupancy between units (units overlap freely) — **observed building log_roll (M3.6): the java reference pushes clustered AIR units apart (air-air collisions resolve every tick), so multi-unit deploys (e.g. minions) drift after spawn where gpusim keeps the cluster overlapped. Single-unit scenarios are unaffected — the log_roll trace is byte-identical. Multi-unit clusters stay outside byte-verified scenarios until this is ported (post-M5, per the M3 checklist)** | OPEN | M2/M5+ |
| 4 | Spells: damage + radius + crown-% core DONE (M3.3, verified); projectile flight time DONE (M3.4); ticking zones DONE (M3.5); **log model DONE (M3.6): the spellAsDeploy two-stage chain (deploy projectile at the cast point + rolling piercing sub-projectile with minDistance gate, single-hit-per-entity, directional pushback 700 and 15 % crown damage) is ported 1:1 and the `log_roll` scenario trace is byte-identical to the java_patched reference (2026-09-15)** | DONE | — |
| 5 | Placement zones + deploy timers enforced via `play()` (M3.2, test_m3); direct `deploy()` API still trusts the caller (tests/scripts only — the policy path is play()) | PARTIAL | M3/M4 |
| 6 | Overtime end uses crown/tower-HP rules (simplified); no per-tick edge cases — **VERIFIED 2026-09-15 (M3.7) against `GameEngine.checkTimeLimit`: because the check runs at step 13 with the frame counter incremented at the end of the tick, the regular-time end (`frame >= 3600`) and the OT end (`frame >= 6000`) land during ticks 3601/6001, i.e. elapsed 180.05 s / 300.05 s; the double-elixir switch (frame >= 2400) only affects the NEXT tick's regen → first x2 tick at 120.10 s, first x3 at 240.10 s. gpusim's boundaries shifted to land on the same ticks (half-tick margin, strict `>` for the elixir phases); pinned by test_m3 (`test_time_limit_boundaries_match_java`, `test_elixir_phase_boundaries_match_java`)** | DONE | — |
| 7 | Card cycle/hand modelled (M3.1: play() rotates played card to the back of the 8-card cycle, test_m3) — DONE; deck-order = hand[0:4] + cycle[4:8] per the reference. **FIXED 2026-09-15 (found by the M4.2 replay diff): the draw queue is 4 slots and `cycle_pos` wrapped with `% 8`, so after the 4th play the next draw read an uninitialised slot (hand slot became -1, rotation lost). Now `% 4`; regression: test_m3.test_full_cycle_draw_order_no_empty_slots. gpusim's draw queue is an exact FIFO equivalent of the Java `Hand.playCard` queue (verified against 8 recorded replays: all recorded hand labels are explained by the model)** | DONE | — |
| 10 | Unit spawn overflow (slots > MAX_UNITS) silently dropped | OPEN | M5 |
| 13 | King tower activation + attacks — RESOLVED (activates on damage/princess loss; 7.0 range, 109 dmg, 1.0s cd) | DONE | — |
| 14 | Projectiles: homing flight + radius impact DONE; remaining: non-homing/arc shots (gravity), AOE-on-impact, scatter/pierce/returning projectiles, tower shots still instant | PARTIAL | M2/M3 |
| 15 | Attack windup / loadTime — RESOLVED (AttackStateMachine port: windup = max(0, cd − load)) | DONE | — |
| 16 | Multiple targets / AOE splash / abilities (charge, dash, reflect, kamikaze, death spawns) not implemented | OPEN | M3 |
| 17 | Attack cadence: the Java reference accumulates windup in float32 and fires one tick late (25-tick cycles vs the intended 24). **RESOLVED 2026-09-15 (M4.2b): gpusim reproduces the reference's float32 late-fire exactly — the duel_knight scenario trace is byte-identical (attack ticks [129, 154, 179, 204, 229, 254, 279, 304] on both sides)** | DONE | — |
| 18 | First-hit model (`first hit = hit time − load time`, load accrue during deploy/move): implemented from the reference's community-documented "secret stats" model. Reference-sim parity confirmed byte-identical (M4.2b, 2026-09-15: sync + deploy + windup timelines match the java_patched traces exactly); the remaining check is against real-game timings (e.g. knight 0.5s) via L2 probes/replays | PENDING-VERIFY | M4/L2 |
| 19 | Same-tick resolution: all attacks of a tick apply, deaths resolve at end of tick — mirror duel (`duel_knight`) ends in a mutual KO on the identical tick. Java source shows the same shape (`gameState.processDeaths()` after combat, no mid-tick alive re-check) — **VERIFIED 2026-09-15 (M4.2b): the java_patched duel_knight trace is byte-identical; both knights die on the same tick (last present 328, KO at 329)** | DONE | — |
| 20 | Overkill: damage could push HP below 0 (tower ran to −68 on the killing blow) — FIXED in M4.1: damage clamps at 0, matching Java `Health.takeDamage` (`current -= min(damage, current)`); covered by test_m4 (`push_left` tower ends at exactly 0.0) | DONE | — |
| 21 | Java shuffles each player's deck at reset (`Hand.java`: `Collections.shuffle`, blue = `Random(seed)`, red = `Random(seed+1)` via `GameSession.reset`). gpusim takes the deck order as given — the caller may shuffle, and training uses a fixed order by design. Replicated bit-exactly for replay work in `fidelity/java_random.py` (verified against 8 recorded replays: the model explains ≥90 % of the recorded hand labels, the remainder being consecutive-duplicate label artifacts in the recorded files) | DONE (documented) | — |
| 22 | Unit-level difference flagged by the M4.2 replay diff (`giant-vs-hog s7011`): blue Minions die ~4–5 steps earlier in gpusim than in the recording, moving the first tower-HP divergence to step 15; air pathing matches (both fly direct), so the suspects are tower-vs-flying-unit targeting/range geometry and shot cadence (#17). **Re-run 2026-09-15 (M4.3): the recording-based diffs are partial by construction (red plays unobservable + label noise in the recorded files); the simulator-vs-simulator gate is byte-exact, so this row is downgraded to a recording-artifact question — no simulator defect identified** | PENDING-VERIFY (low) | M5+ |
| 23 | Buff system not modelled (speedMultiplier / Freeze / stacking). Zone tick DAMAGE is java-verified (poison/earthquake scenarios: identical ticks and amounts), but units inside a zone keep full movement speed in gpusim where the reference applies the buff slow (poison −15 %, earthquake −50 %, freeze −100 %). Freeze's slowing is its main in-game effect and is entirely absent. | OPEN (documented) | M5+/L2 |

Resolved: #2 combat (melee/ranged, cooldowns, deaths) — DONE 2026-09-15;
#8 speed formula (`speed * 1000 / 60` game-units/s, from GameUnits.java) — VERIFIED 2026-09-15;
#9 elixir phases (double at 120s, triple at 240s) — DONE; #11/#12 data rulings applied;
#13 king activation + king attacks — DONE; #15 windup/loadTime (AttackStateMachine port) — DONE.

### M4.2b fixes (2026-09-15) — the scenario set is now fully byte-equivalent
* Crown tower collision radius 1.5 → 1.4 (`Tower.CROWN_COLLISION_RADIUS = 1400`,
  `gpusim/env.py`). Symptom: units stopped ~0.1 tile early at max attack range
  (push_left: musketeer froze at d=7.96 vs the reference 7.86 from the king).
* Multi-spawn stagger: the float32-stored 0.1 s delay made `ceil(0.1/0.05)` land
  on 3 ticks (Java: `0.1f = exactly 2 * 0.05f`); fixed in `_stagger_ticks`
  (4-decimal round) plus a half-tick snap against the float32 decrement residue
  in the spawn fire condition. Minions: ticks 21/23/25. Skeletons: no
  `summonDeployDelay` in the data → all three at once (`Card.java`).
* Trace `t` written from the tick index (was the fp32 sim clock → 0.001 s diff
  from tick 1031 onward). All five `java_patched` traces are now `cmp`-identical.

### M3.6 fixes (2026-09-15) — log_roll is byte-equivalent
* Deploy formation offsets (`TroopFactory.createTroop` + `DeployFormation`): the
  data offsets are blue-side in the java frame (+y forward, +x right); red
  mirrors both axes. This sim is y-flipped vs java, so `deploy()` now applies
  `(dx, -dy)` for blue and `(-dx, +dy)` for red. Found by the first log_roll
  attempt: the reference spawned red minions at the mirrored offsets (spawn
  ticks and positions then matched exactly).
* Trace unit names: gpusim emitted the CARD name ("minions"); the reference
  runner emits the UNIT name ("minion"). `_serialize_env` now writes the unit
  name with the java-side normalization (`[^a-z0-9]` stripped, lowercase).
* The Log chain itself: spellAsDeploy deploy projectile (cast point + 3 game
  units forward via the legacy `round(minDistance/1000)`), rolling piercing
  sub-projectile (float32 travel accumulation with the 2.5-tile minDistance
  gate, hit-once per entity, directional pushback 700 for 0.5 s, 15 % crown
  damage) — 1:1 from `SpellFactory` + `ProjectileHitProcessor` +
  `PiercingHitDetector` (+ `KnockbackHelper` / `CombatSystem` knockback rules).

Empty table = the sim is fully equivalent (M4 gate).
