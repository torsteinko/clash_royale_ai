# New cards (heroes, evolutions, fresh releases like Minion Giant)

Supercell ships new cards/evolutions/heroes faster than any dataset updates.
The rule: **never silently ignore a new card** — an invisible win condition on
the opponent's side corrupts the game state the policy sees.

## Where new cards break things

1. **Detector taxonomy** — a unit with no class gets either missed or
   mislabeled as a similar card (worse than missed: state lies).
2. **Hand matching** (`data/card_templates/`) — if *you* play it, its slot
   can never lock, blocking cycle tracking for that slot.
3. **Registry** (`data/decks/cards_api.json`) — elixir costs + names for the
   policy's card embedding. Verified Sept 2026: 121 items, 4 heroes; already
   includes recent cards (Boss Bandit, Spirit Empress, Rune Giant) but NOT
   "Minion Giant" (released days ago).

## Onboarding checklist for a new card

1. **Registry**: add the card (name, elixir cost, type) to
   `data/decks/cards_api.json`. Ideally refresh from the official API with a
   token (`https://api.clashroyale.com/v1/cards`) — script to be added.
2. **Hand template**: drop the card art into `card_images/base/<name>.png`
   and run `tools/convert_card_images_to_templates.py` (or harvest a slot
   crop from a frame with `tools/extract_card_template_from_frame.py`).
   Names must be underscore-style (`minion_giant`).
3. **In-game capture**: let the card appear in captures (opponent plays it or
   you do); collect ~50–200 frames with it.
4. **Label + incremental fine-tune**: label the crops (they can be
   pre-detected by the current model) and fine-tune the detector for a few
   epochs. Cheap on the 5070 Ti; keep the previous weights for A/B.
5. **Verify**: `python scripts/analyze_recording.py <recording> --save-debug 10`
   and check the card is detected with the right team (tint) in both halves.

## Interim handling (before the above is done)

- Evolutions (39) and heroes (4) already have templates + taxonomy entries.
- Unknown units: the 296-class model will usually mislabel them — treat
  low-confidence detections in unexpected places as "unknown unit" in any
  manual review; do not trust the label blindly.
- Monitor: after each game patch (balance + new cards), re-run the analyzer
  on a fresh recording and compare per-class detection counts week over week.
