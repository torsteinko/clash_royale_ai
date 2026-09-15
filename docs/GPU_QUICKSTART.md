# GPU quickstart — run the batched sim on your 5070 Ti

One linear list. Run from **PowerShell** (or the Command Prompt). ~10 minutes.

1. **Installer Python (hvis du ikke har det):** `winget install Python.Python.3.12` — lukk og åpne terminalen på nytt etterpå.
2. **Hent repoet (hvis ikke allerede klonet):**
   `git clone --branch meidell/linux-state-pipeline https://github.com/torsteinko/clash_royale_ai.git`
   Har du det fra før: `cd clash_royale_ai` → `git fetch origin` → `git checkout meidell/linux-state-pipeline` → `git pull`.
3. **Lag venv og aktiver den:**
   `py -3.12 -m venv .venv` deretter `.venv\Scripts\Activate.ps1`
4. **Installer PyTorch med Blackwell-støtte (RTX 50-serien krever CUDA 12.8-bygget):**
   `pip install torch --index-url https://download.pytorch.org/whl/cu128`
   *(Feiler dette: prøv `cu126` i stedet for `cu128` — men for 5070 Ti er cu128 riktig.)*
5. **Verifiser GPU-en:**
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   → skal skrive `True NVIDIA GeForce RTX 5070 Ti`.
6. **Kjør GPU-benchen (fra repo-roten):**
   `python -m gpusim.tests.bench_gpu`
   → skriver ut f.eks. `B=4096: 200 ticks in ...s = ... env-steps/s (cuda)`

**Send meg tallet!** Det er den første ekte 5070 Ti-målingen vår.

## Hva du ser på

Sim-en kjører 4096 parallelle kamper samtidig: en ridder som marsjerer mot en
musketeer som skyter prosjektiler — targeting, kamp, flytid og tårn-logikk
evalueres hver tick. Prøv gjerne større batch: `python -m gpusim.tests.bench_gpu 16384 300`.

## Troubleshooting

- `torch.cuda.is_available() == False` → feil hjul installert; kjør pip-kommandoen i steg 4 på nytt (den *skal* laste ned en ~2,5 GB CUDA-versjon).
- `no kernel image is available for execution` → sm_120 uten støtte; du har et gammelt torch → oppgrader: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128`.
- `ModuleNotFoundError: gpusim` → du står ikke i repo-roten. `cd` dit repoet ligger.
