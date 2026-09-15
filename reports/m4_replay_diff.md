# M4.2 replay diff — recorded Java replays replayed in gpusim

generated 2026-09-15 19:58:19 · data dir `fidelity/patched` · tolerances: tower HP ±1.0, elixir ±1.5

| replay | steps | blue acts | labels model-match | red acts | red skipped | tower-HP max diff | first div. | be max | re max |
|---|---|---|---|---|---|---|---|---|---|
| 0915-080002_hog-vs-giant_rule_based_s7000.json | 241 | 35 | 35/36 | 4 | 24 | 3584.0 | 9 | 2.054 | 8.321 |
| 0915-080228_hog-vs-xbow_rule_based_s7002.json | 401 | 75 | 75/86 | 4 | 31 | 4824.0 | 35 | 3.25 | 10.0 |
| 0915-080232_hog-vs-golem_rule_based_s7006.json | 241 | 33 | 33/41 | 18 | 18 | 2370.58 | 28 | 2.572 | 8.839 |
| 0915-080234_hog-vs-mortar_rule_based_s7008.json | 241 | 23 | 28/53 | 3 | 17 | 4824.0 | 6 | 9.0 | 10.0 |
| 0915-080236_giant-vs-hog_rule_based_s7011.json | 200 | 20 | 20/20 | 1 | 1 | 1895.444 | 15 | 1.304 | 7.0 |
| 0915-080238_xbow-vs-hog_rule_based_s7013.json | 224 | 33 | 33/39 | 2 | 0 | 4746.0 | 17 | 1.161 | 7.839 |
| 0915-080239_golem-vs-hog_rule_based_s7014.json | 366 | 49 | 53/55 | 0 | 28 | 3044.0 | 10 | 10.0 | 10.0 |
| 0915-120343_hog-vs-giant_rule_based_s7000.json | 241 | 36 | 40/42 | 4 | 20 | 4824.0 | 12 | 6.0 | 8.321 |

## 0915-080002_hog-vs-giant_rule_based_s7000.json

- blue deck `hog` vs red deck `giant` (seed 7000), recorded result **win**, 241 steps, replayed in 11.8s
- recorded blue labels explained by the shuffle+FIFO hand model: 35/36 — failures: [{"step": 124, "label": "Cannon", "model_hand": ["fireball", "hogrider", "musketeer", "zap"]}]
- tower-HP diff: max 3584.0, first over tol at step 9, steps over tol 233
- elixir diff: blue max 2.054, red max 8.321
- final sim state: winner 0, crowns [5, 0], towers alive [True, True, True, False, False, False], units alive 23
- blue-side flags: [{"step": 124, "kind": "blue_label_unresolved", "detail": "Cannon"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-deck': 14, 'not-in-red-hand': 10}

## 0915-080228_hog-vs-xbow_rule_based_s7002.json

- blue deck `hog` vs red deck `xbow` (seed 7002), recorded result **win**, 401 steps, replayed in 19.2s
- recorded blue labels explained by the shuffle+FIFO hand model: 75/86 — failures: [{"step": 58, "label": "Skeletons", "model_hand": ["log", "musketeer", "fireball", "zap"]}, {"step": 59, "label": "Skeletons", "model_hand": ["log", "musketeer", "fireball", "zap"]}, {"step": 153, "label": "Skeletons", "model_hand": ["log", "musketeer", "fireball", "zap"]}]
- tower-HP diff: max 4824.0, first over tol at step 35, steps over tol 367
- elixir diff: blue max 3.25, red max 10.0
- final sim state: winner 0, crowns [5, 0], towers alive [True, True, True, False, False, False], units alive 49
- blue-side flags: [{"step": 57, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}, {"step": 58, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}, {"step": 152, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}, {"step": 153, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}, {"step": 185, "kind": "blue_play_rejected", "detail": "Musketeer @(9.0,29.0)"}, {"step": 205, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}, {"step": 207, "kind": "blue_play_rejected", "detail": "Musketeer @(9.0,29.0)"}, {"step": 256, "kind": "blue_play_rejected", "detail": "Skeletons @(9.0,29.0)"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-hand': 31}

## 0915-080232_hog-vs-golem_rule_based_s7006.json

- blue deck `hog` vs red deck `golem` (seed 7006), recorded result **win**, 241 steps, replayed in 11.6s
- recorded blue labels explained by the shuffle+FIFO hand model: 33/41 — failures: [{"step": 40, "label": "Cannon", "model_hand": ["fireball", "musketeer", "zap", "log"]}, {"step": 41, "label": "Cannon", "model_hand": ["fireball", "musketeer", "zap", "log"]}, {"step": 90, "label": "Cannon", "model_hand": ["fireball", "musketeer", "zap", "log"]}]
- tower-HP diff: max 2370.58, first over tol at step 28, steps over tol 214
- elixir diff: blue max 2.572, red max 8.839
- final sim state: winner -1, crowns [1, 1], towers alive [True, True, False, True, False, True], units alive 10
- blue-side flags: [{"step": 40, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 41, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 90, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 137, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 138, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 139, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 220, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 221, "kind": "blue_label_unresolved", "detail": "Cannon"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-deck': 18}

## 0915-080234_hog-vs-mortar_rule_based_s7008.json

- blue deck `hog` vs red deck `mortar` (seed 7008), recorded result **loss**, 241 steps, replayed in 10.7s
- recorded blue labels explained by the shuffle+FIFO hand model: 28/53 — failures: [{"step": 17, "label": "HogRider", "model_hand": ["cannon", "icespirits", "fireball", "zap"]}, {"step": 31, "label": "Musketeer", "model_hand": ["cannon", "skeletons", "fireball", "zap"]}, {"step": 51, "label": "Log", "model_hand": ["cannon", "hogrider", "fireball", "zap"]}]
- tower-HP diff: max 4824.0, first over tol at step 6, steps over tol 236
- elixir diff: blue max 9.0, red max 10.0
- final sim state: winner 0, crowns [4, 0], towers alive [True, True, True, False, False, True], units alive 4
- blue-side flags: [{"step": 16, "kind": "blue_play_rejected", "detail": "HogRider @(14.5,25.5)"}, {"step": 30, "kind": "blue_play_rejected", "detail": "Musketeer @(9.0,29.0)"}, {"step": 51, "kind": "blue_label_unresolved", "detail": "Log"}, {"step": 55, "kind": "blue_play_rejected", "detail": "HogRider @(14.5,25.5)"}, {"step": 56, "kind": "blue_play_rejected", "detail": "HogRider @(14.5,25.5)"}, {"step": 71, "kind": "blue_play_rejected", "detail": "Cannon @(9.0,29.0)"}, {"step": 107, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 108, "kind": "blue_label_unresolved", "detail": "Cannon"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-hand': 12, 'not-in-red-deck': 5}

## 0915-080236_giant-vs-hog_rule_based_s7011.json

- blue deck `giant` vs red deck `hog` (seed 7011), recorded result **win**, 200 steps, replayed in 8.3s
- recorded blue labels explained by the shuffle+FIFO hand model: 20/20
- tower-HP diff: max 1895.444, first over tol at step 15, steps over tol 186
- elixir diff: blue max 1.304, red max 7.0
- final sim state: winner 0, crowns [5, 0], towers alive [True, True, True, False, False, False], units alive 5
- red plays not replayed (unobservable/unmodelled): {'not-in-red-hand': 1}

## 0915-080238_xbow-vs-hog_rule_based_s7013.json

- blue deck `xbow` vs red deck `hog` (seed 7013), recorded result **win**, 224 steps, replayed in 10.9s
- recorded blue labels explained by the shuffle+FIFO hand model: 33/39 — failures: [{"step": 51, "label": "Archer", "model_hand": ["log", "fireball", "knight", "xbow"]}, {"step": 93, "label": "Archer", "model_hand": ["log", "fireball", "knight", "xbow"]}, {"step": 139, "label": "Archer", "model_hand": ["log", "fireball", "knight", "xbow"]}]
- tower-HP diff: max 4746.0, first over tol at step 17, steps over tol 208
- elixir diff: blue max 1.161, red max 7.839
- final sim state: winner -1, crowns [1, 0], towers alive [True, True, True, True, False, True], units alive 14
- blue-side flags: [{"step": 50, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}, {"step": 92, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}, {"step": 138, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}, {"step": 170, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}, {"step": 193, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}, {"step": 216, "kind": "blue_play_rejected", "detail": "Archer @(9.0,29.0)"}]

## 0915-080239_golem-vs-hog_rule_based_s7014.json

- blue deck `golem` vs red deck `hog` (seed 7014), recorded result **win**, 366 steps, replayed in 15.1s
- recorded blue labels explained by the shuffle+FIFO hand model: 53/55 — failures: [{"step": 27, "label": "Tornado", "model_hand": ["lightning", "babydragon", "golem", "darkwitch"]}, {"step": 42, "label": "Tornado", "model_hand": ["lightning", "babydragon", "golem", "minions"]}]
- tower-HP diff: max 3044.0, first over tol at step 10, steps over tol 357
- elixir diff: blue max 10.0, red max 10.0
- final sim state: winner 0, crowns [5, 0], towers alive [True, True, True, False, False, False], units alive 64
- blue-side flags: [{"step": 27, "kind": "blue_label_unresolved", "detail": "Tornado"}, {"step": 28, "kind": "blue_play_rejected", "detail": "DarkWitch @(5.5,14.5)"}, {"step": 42, "kind": "blue_label_unresolved", "detail": "Tornado"}, {"step": 50, "kind": "blue_label_unresolved", "detail": "Minions"}, {"step": 62, "kind": "blue_label_unresolved", "detail": "Skeletons"}, {"step": 69, "kind": "blue_label_unresolved", "detail": "Valkyrie"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-hand': 28}

## 0915-120343_hog-vs-giant_rule_based_s7000.json

- blue deck `hog` vs red deck `giant` (seed 7000), recorded result **loss**, 241 steps, replayed in 11.1s
- recorded blue labels explained by the shuffle+FIFO hand model: 40/42 — failures: [{"step": 68, "label": "IceSpirits", "model_hand": ["fireball", "musketeer", "zap", "cannon"]}, {"step": 202, "label": "Cannon", "model_hand": ["fireball", "hogrider", "skeletons", "musketeer"]}]
- tower-HP diff: max 4824.0, first over tol at step 12, steps over tol 230
- elixir diff: blue max 6.0, red max 8.321
- final sim state: winner 0, crowns [5, 1], towers alive [True, True, False, False, False, False], units alive 14
- blue-side flags: [{"step": 67, "kind": "blue_play_rejected", "detail": "IceSpirits @(9.0,20.5)"}, {"step": 202, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 203, "kind": "blue_play_rejected", "detail": "HogRider @(7.5,10.5)"}, {"step": 224, "kind": "blue_label_unresolved", "detail": "Cannon"}, {"step": 234, "kind": "blue_label_unresolved", "detail": "Skeletons"}, {"step": 237, "kind": "blue_label_unresolved", "detail": "IceSpirits"}]
- red plays not replayed (unobservable/unmodelled): {'not-in-red-deck': 10, 'not-in-red-hand': 10}
