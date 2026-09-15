# Arbeidsliste mot GPU-trening — KJØR ALLE SPOR PARALLELT

Mål: hele treningsstacken kjører på GPU-simulatoren (`gpusim/`), validert mot Java-referansen.

**VIKTIG: M3-, M4- og M5-oppgavene er UAVHENGIGE — vent ikke på M3-detaljene.**
Følg PRIORITET-rekkefølgen under (kritisk sti først), kryss av etter hvert.

## PRIORITET (kritisk sti)
1. M4.1 scenario-runner (gpusim-siden + trace-format)
2. M5.1b SB3-innpakning (gymnasium.Env + MaskablePPO smoke på CPU)
3. M5.2 train_gpu.py (PPO-headless på gpusim)
4. M4.2/4.3 replay-diff + fidelity-rapport
5. M3.4–M3.8 (spell-flytid, tick-soner, log, OT, fuzz)
6. M5.3/5.4 bench + sluttverifisering

## Status
- [x] M3.1 Kort-syklus/hånd (d7f31ab) — + FIFO-fiks av trekk-køen 15/9 (M4.2-replay-diff fant at håndslot ble -1 etter 4. spill; test_m3-regresjon)
- [x] M3.2 Deploy-soner
- [x] M3.3 Spells kjerne (crown-%, fireball 688→206 verifisert)
- [x] M5.0 VecEnv v0 (b821a0e): obs 421, masks, step, determinisme — 3 tester
- [x] M4.1 Scenario-runner + trace/rapport-format (fidelity/m4_scenarios.py, docs/M4_TRACE_FORMAT.md; 8 nye tester i test_m4)
- [x] M5.1b SB3-innpakning: gymnasium.Env rundt GPUSimVecEnv + MaskablePPO-smoke (2000 steg CPU, SMOKE OK; 8 tester i test_sb3)
- [x] M5.2 train_gpu.py: headless PPO på batchet sim med auto-reset per kamp, progress.csv + TensorBoard, sjekkpunkter + resume (verifisert fresh+resume på CPU)
- [x] M4.2 Replay-diff: fidelity/m4_replay_diff.py kjører 8 recorded replays i gpusim (java-shuffle + FIFO-hånd modellert), diff tårn-HP/elixir; fant og fikset kort-syklusbuggen; rapport i reports/m4_replay_diff.* + funn i reports/m4_replay_findings.md

## Oppgaver (første ukryssede øverst = neste å ta)
- [x] M4.1 Scenario-runner (fidelity/m4_scenarios.py + docs/M4_TRACE_FORMAT.md): 5 faste scenarios (duell_knight/duell_musketeer, tårnpress, spell-hit, push) → per-tick trace (tower-HP/elixir/tid/units) + manifest + rapportformat (reports/m4_scenarios_report.{md,json}); to fulle kjøringer byte-identiske; diff-verktøy (`compare`) klart for Java-traces. Bonus-fiks: HP-gulv på 0 ved overkill (Java Health.takeDamage-semantikk)
- [x] M5.1b SB3-innpakning: `gpusim/sb3_env.py` (GPUSimGymEnv med action_masks + make_sb3_vec_env) + `gpusim/train_sb3_smoke.py` (~2000 steg, CPU: MaskablePPO, greedy-handlinger lovlige under masks, modell lagres) — 8 tester i test_sb3
- [x] M5.2 train_gpu.py: `gpusim/train_gpu.py` — MaskablePPO på GPUSimSB3VecEnv (ÉN batchet lockstep-sim, per-kamp auto-reset via partial reset), fast deck + scripted motstander (random/passive), progress.csv + TensorBoard (valgfritt), ckpt_<steg>.zip + latest.zip hver --ckpt-every, `--resume auto` verifisert (fresh 1024→2048 steg, resume til 4096 på CPU; ~40-50 steg/s på 2-vCPU-boksen)
- [x] M4.2 Replay-diff: `fidelity/m4_replay_diff.py` (`run --dir fidelity/replays`): blå handlinger 1:1 på riktig steg, rød rekonstruert fra spawn-evidens + elixir-budsjett (spells/ability-spawns flagges), diff av tårn-HP (6 slots) og elixir per steg. Verifisert på 8 replays; fant kort-syklusbuggen (fikset) og java-shuffle-modellen (fidelity/java_random.py); funn i reports/m4_replay_findings.md
- [ ] M4.2b Scenario-diff mot Java: kjør scenarios på clash-training (gym-bridge, ticks_per_step=1, docs/M4_TRACE_FORMAT.md §3) → `compare` java vs gpusim-traces. NB: Java-shuffler hånden ved reset → velg hand-slot på kortnavn ved kjøring (fidelity/java_random.py har shuffle-replikatet); javaside-runner gjenstår
- [ ] M4.3 Fidelity-rapport i reports/ med terskler og kjente avvik (bruk M4.2-funnene: #22 må undersøkes per-tick)
- [ ] M3.4 Spell-prosjektil-flytid (fireball/arrows: fly til punkt → deretter AOE)
- [ ] M3.5 Tikkende soner: poison/earthquake (lifeDuration + hitSpeed + damage per tick)
- [ ] M3.6 Log (spellAsDeploy rullende prosjektil eller presis tilnærming dokumentert i DIVERGENCES)
- [ ] M3.7 OT/elixir-grenser verifisert mot Java checkTimeLimit + test
- [ ] M3.8 Fuzz: 200 fullverdige kamper via play() fra begge sider uten krasj + determinisme-test
- [ ] M5.3 CPU-bench oppdatert + docs/GPU_QUICKSTART.md steg-for-steg verifisert (legg til trenings-steg: pip install stable-baselines3 sb3_contrib, python -m gpusim.train_gpu)
- [ ] M5.4 SLUTTVERIFISERING: alle suiter grønne + bench + docs → kort sluttrapport til Olsen («NÅ kan du kjøre»)

## Utsatt (dokumenteres i DIVERGENCES.md, ikke blokkerende)
- [ ] Enhets-kollisjon/okkupering (#1/#3) — etter M5 hvis toleransene holder
- [ ] Ikke-homing/gravitasjons-prosjektiler (#14), scatter/pierce/return (#16)
- [ ] Evulusjoner/heroes-egenskaper (utenfor pool-deckene først)

## Testsviter (alle MÅ være grønne før commit)
test_smoke · test_combat · test_projectiles · test_pathing · test_m3 · test_vecenv · test_m4 · test_sb3
(kjør alle: `for t in test_smoke test_combat test_projectiles test_pathing test_m3 test_vecenv test_m4 test_sb3; do python3 -m gpusim.tests.$t || break; done` — 45 tester)
