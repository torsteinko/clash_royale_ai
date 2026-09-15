# Arbeidsliste mot GPU-trening — fortsett til ALT er krysset av

Mål: hele treningsstacken kjører på GPU-simulatoren (`gpusim/`), validert mot Java-referansen.
Regler: porter 1:1 fra Java, aldri juks med tester, oppdater DIVERGENCES ved hver endring.

Status: M0–M2 ferdig (20 tester). M3 delvis (kort-syklus, spells-grunnmur, deploy-soner).

## M3 — regler og spells (ferdigstill)
- [x] M3.1 Kort-syklus/hånd: set_deck + play() med rotasjon — test_m3
- [x] M3.2 Deploy-soner: egen halvdel + lomme etter prinsesse-fall — test_m3
- [x] M3.3 Spells kjerne: instant AOE + crown-tower-% (fireball 688→206 på tårn) — test_m3
- [ ] M3.4 Spell-prosjektil-flytid (fireball/arrows): flygende spell-prosjektil til punkt, deretter AOE (gravity/pushback kan utsettes — dokumenter)
- [ ] M3.5 Tikkende soner: poison/earthquake (lifeDuration + hitSpeed, damage per tick)
- [ ] M3.6 Log: spellAsDeploy-rullende prosjektil — eller presis tilnærming dokumentert i DIVERGENCES
- [ ] M3.7 OT/elixir-grenser verifisert mot Java checkTimeLimit (dobbel 120s, trippel 240s, tiebreak) + test
- [ ] M3.8 Fuzz: 200 fullverdige kamper via play() fra begge sider (tilfeldige lovlige trekk) uten krasj + determinisme-test

## M4 — differensial-harness (porten til «validert ekvivalent»)
- [ ] M4.1 Scenario-runner: identiske scenarios i Java-sim og gpusim (duell, tårnpress, spell-hit); sammenlign HP/elixir/tid innen toleranse (±2 %, ±2 ticks). Rapporter avvik.
- [ ] M4.2 Replay-diff: egne replay-logger (training/record_replays.py-format) → kjør handlingene i gpusim; diff tårn-HP/elixir. Skriv harness i fidelity/ eller tools/.
- [ ] M4.3 Fidelity-rapport i reports/ med terskler og kjente avvik

## M5 — GPU-klar (integrasjon)
- [ ] M5.1 VecEnv-adapter (gpusim/vecenv.py): SB3-kompatibel reset/step med obs (dokumenter obs-layout; gjenbruk Java-obs der mulig)
- [ ] M5.2 Treningsskript train_gpu.py: PPO-headless på gpusim (samme hyperparams som Java-stacken der relevant)
- [ ] M5.3 Bench-tall på CPU oppdatert + docs/GPU_QUICKSTART.md verifisert steg-for-steg
- [ ] M5.4 SLUTTVERIFISERING: alle suiter grønne + bench + docs oppdatert → kort sluttrapport til Olsen («NÅ kan du kjøre»)

## Blokkert / utsatt (dokumenter i DIVERGENCES.md)
- [ ] Enheter-fysisk kollisjon/okkupering (#1/#3) — kan utsettes til etter M4 hvis toleransene holder
- [ ] Ikke-homing prosjektiler/gravitasjonsbuer (#14), scatter/pierce/return (#16 abilities) — etter M5 hvis pool-deckene ikke krever dem
