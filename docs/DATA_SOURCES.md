# Fidelity data sources (P0, Sep 15 2026)

Goal: authoritative, current Clash Royale card/mechanics data to (a) fix crforge's
card constants, (b) feed the GPU simulator's card tables.

## Findings

| Source | What | Freshness | Verdict |
|---|---|---|---|
| `RoyaleAPI/cr-api-data` (GitHub + GH Pages) | JSON mirror of Supercell's csv_logic: full stat tables per character/spell/building/projectile incl. per-level arrays | **Stale: last commit Oct 2023** | Use as SCHEMA/format reference + baseline diff; not current |
| `smlbiobot/cr-csv` | Raw `assets/csv_logic/*.csv` dump | Stale: last push Aug 2023 | Format reference only |
| `royaleapi.com/card/<name>` | Live values (continuously updated site) | Current | Cloudflare-blocked from datacenter IPs; works from a residential browser (Olsen). Spot checks only |
| Official Supercell API `/v1/cards` | Card list + maxLevel | Current | Insufficient (no HP/damage stats) |
| **APK `assets/csv_logic/`** | Supercell's own data tables (characters, spells, buildings, projectiles, areas, ...) | **Always current (per build)** | **Ground truth. Owner supplies the APK extract; refresh after each balance update.** |

## APK extraction recipe (owner, Windows)

1. Download latest Clash Royale APK (APKMirror or APKPure; prefer the plain `.apk`
   over `.apkm/.xapk` bundles for easiest extraction).
2. Rename `Clash.Royale.*.apk` -> `clash.zip` (an APK is just a zip).
3. Right-click -> Extract All (or open with 7-Zip).
4. In the extracted folder: `assets/csv_logic/` (the data tables). Optional:
   `assets/csv_client/`.
5. Zip the `csv_logic` folder (ends up a few MB) and send it over. Discord
   attachment is fine; otherwise Google Drive link.

## Pipeline (to build)

1. `csv_logic` parser -> normalized JSON (schema modelled on cr-api-data's
   `cards_stats_*.json` so comparisons/diffs are trivial).
2. Diff tool: parser output <-> crforge `data/cards.json` <-> cr-api-data (2023)
   -> fidelity report (per card, per field).
3. Regenerate crforge card data from the current extraction; re-run the fidelity
   probes (training/fidelity_probe*.py) and the differential harness for the GPU sim.
4. Refresh loop: re-extract after each balance update; keep versioned snapshots
   in the repo (small JSONs).
