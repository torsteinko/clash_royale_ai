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
- [x] M3.1 Kort-syklus/hånd (d7f31ab)
- [x] M3.2 Deploy-soner
- [x] M3.3 Spells kjerne (crown-%, fireball 688→206 verifisert)
- [x] M5.0 VecEnv v0 (b821a0e): obs 421, masks, step, determinisme — 3 tester

## Oppgaver (første ukryssede øverst = neste å ta)
- [ ] M4.1 Scenario-runner: faste scenarios (duell, tårnpress, spell-hit) kjørt i gpusim med maskinlesbar trace (tower-HP/elixir/tid per tick) → fidelity/m4_scenarios.py + rapportformat; Java-siden kobles på etterpå (gym-bridge på clash-training eller manuelt script)
- [ ] M5.1b SB3-innpakning: gymnasium.Env-subklasse rundt GPUSimVecEnv + MaskablePPO smoke-run (~2000 steps, CPU) i training/ eller gpusim/
- [ ] M5.2 train_gpu.py: PPO-headless på gpusim (self-play eller fast deck), logging + checkpointing + resume
- [ ] M4.2 Replay-diff: les replay-logger (training/record_replays.py-format) → kjør handlingene i gpusim → diff tårn-HP/elixir
- [ ] M4.3 Fidelity-rapport i reports/ med terskler og kjente avvik
- [ ] M3.4 Spell-prosjektil-flytid (fireball/arrows: fly til punkt → deretter AOE)
- [ ] M3.5 Tikkende soner: poison/earthquake (lifeDuration + hitSpeed + damage per tick)
- [ ] M3.6 Log (spellAsDeploy rullende prosjektil eller presis tilnærming dokumentert i DIVERGENCES)
- [ ] M3.7 OT/elixir-grenser verifisert mot Java checkTimeLimit + test
- [ ] M3.8 Fuzz: 200 fullverdige kamper via play() fra begge sider uten krasj + determinisme-test
- [ ] M5.3 CPU-bench oppdatert + docs/GPU_QUICKSTART.md steg-for-steg verifisert
- [ ] M5.4 SLUTTVERIFISERING: alle suiter grønne + bench + docs → kort sluttrapport til Olsen («NÅ kan du kjøre»)

## Utsatt (dokumenteres i DIVERGENCES.md, ikke blokkerende)
- [ ] Enhets-kollisjon/okkupering (#1/#3) — etter M5 hvis toleransene holder
- [ ] Ikke-homing/gravitasjons-prosjektiler (#14), scatter/pierce/return (#16)
- [ ] Evulusjoner/heroes-egenskaper (utenfor pool-deckene først)

## Testsviter (alle MÅ være grønne før commit)
test_smoke · test_combat · test_projectiles · test_pathing · test_m3 · test_vecenv
