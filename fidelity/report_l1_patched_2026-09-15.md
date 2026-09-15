# L1 fidelity report (card level): crforge vs noff.gg live dump (2026-09)

- crforge base cards: 130 (+110 hero/evo variants, excluded)
- reference cards: 127 | card-matched: 110
- stat mismatches: **6** | elixir-cost mismatches: **2**
- level basis: reference = **level 1** (verified: noff base values == 2023 per-level array index 0 for Knight/Giant/Musketeer/Hog Rider); crforge units.json = level-1 base values as well. All comparisons are level-1 vs level-1.
- manual corrections applied (reference overridden): 1 (see reference/corrections.json)

## Stat mismatches

| card | field | crforge | reference | delta |
|---|---|---|---|---|
| MinionHorde | damage | 42 | 46 | -4 |
| SkeletonArmy | health | 32 | 51 | -19 |
| SkeletonArmy | damage | 32 | 51 | -19 |
| SkeletonArmy | attackCooldown | 1.1 | 1 | +0.1 |
| Xbow | damage | 17 | 36 | -19 |
| Xbow | attackCooldown | 0.3 | 0.4 | -0.1 |

## Elixir cost mismatches

| card | crforge | reference |
|---|---|---|
| DarkMagic | 3 | 5 |
| GoblinPartyHut | 5 | 4 |

## In reference but missing from crforge (18)

Bandit, Cannoneer, Dagger Duchess, Elite Barbarians, Flying Machine, Furnace, Giant Snowball, Guards, Heal Spirit, Ice Golem, Lumberjack, Magic Archer, Minion Giant, Ronin, Royal Chef, Sparky, The Log, Tower Princess

## In crforge but not in reference (20)

AngryBarbarians, Assassin, BalloonBomb, BarbarianLauncher, BombTowerBomb, DarkElixir_Bottle, DartBarrell, EliteArcher, ElixirBarrel, FirespiritHut, GiantSkeletonBomb, Heal, IceGolemite, Log, RageBarbarian, RageBarbarianBottle, SkeletonBalloon, SkeletonWarriors, Snowball, ZapMachine