# CCNSA — Conservative-Critic Neuro-Symbolic Agent

Reference implementation and reproducibility package for:

> Uday Pratap Singh, Bersha Kumari. **"CCNSA: A Neuro-Symbolic Reinforcement Learning Agent for Network Trauma Recovery with Faithful Explanations."** (manuscript under review)

CCNSA integrates a twin-critic Soft Actor-Critic backbone, a CQL-style conservative critic penalty, a PC-algorithm causal-discovery layer, an eight-rule differentiable symbolic layer, and a behavioural-cloning warm-start, evaluated on a simulated network-trauma-recovery environment (Track A) and the real UNSW-NB15 intrusion-detection dataset (Track B).

This repository contains **all code needed to reproduce every number reported in the paper** — training, baselines, ablations, significance tests, and figure generation. It does **not** contain pre-computed results, model checkpoints, or the manuscript itself; every number in the paper was produced by running this code and is reproducible from a fresh clone (seeds are fixed and logged — see "Reproducibility" below).

## Contents

```
ccnsa_revised.py     # main script: environment, agents, baselines, training,
                      # ablations, statistical tests, all --track entry points
gen_fig0.py           # generates the architecture/graphical-abstract figure
gen_figs.py           # generates the results figures (seed distribution,
                       # final performance, ablation, faithfulness, radar,
                       # FLOPs/latency, advantage profile, Track B)
requirements.txt
LICENSE               # Apache 2.0
CITATION.cff
```

`ccnsa_revised.py` bootstraps its own dependencies on first run (`pip install --break-system-packages` for anything missing), so `requirements.txt` is provided for reference / pinning in CI rather than as a mandatory pre-install step.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for anything beyond `--quick` smoke tests. All reported wall-clock numbers in the paper (e.g. Track A: ~10 hours full run; individual ablation sweeps: ~1.5–2.5 hours) were measured on a single Quadro RTX 8000 (48 GB). CPU-only runs work but are substantially slower (see the CPU-latency figures in the paper's Limitations section for a sense of the per-step overhead).
- ~2 GB disk for the UNSW-NB15 cache (downloaded automatically on first Track B run from the HuggingFace mirror below).

## Quick start

```bash
git clone https://github.com/singhudaypratap/CCNSA.git
cd CCNSA
pip install -r requirements.txt        # optional -- the script self-installs what's missing
python ccnsa_revised.py --track AB --quick   # smoke test, minutes not hours
```

If the smoke test runs clean, drop `--quick` for a full, paper-scale run.

## All `--track` options

| Track | What it does | Approx. runtime (RTX 8000, full seeds) |
|---|---|---|
| `A` | Track A only: 9-agent simulator comparison | ~8–10 hr |
| `B` | Track B only: UNSW-NB15 causal discovery + audit rules + classifier context (downloads dataset automatically) | ~30–60 min |
| `AB` (default) | Both tracks | ~9–11 hr |
| `ABL` | Ablation-only re-run (skips Track A/B training+eval) — for iterating on `ABLATION_VARIANTS` without re-paying full pipeline cost | ~2–3 hr |
| `RULES` | Symbolic-rule-count sensitivity sweep (r=4/8/16) | ~1.5–2 hr |
| `CAUSAL` | Causal-discovery method sweep (PC vs NOTEARS vs DirectLiNGAM) on downstream Track-A performance | ~1.5–2 hr |
| `CASCADE` | 4-arm core ablation re-run on `CascadingTraumaEnvironment` (a task variant with a fixed node dependency hierarchy + power budget, designed to test whether the causal/symbolic layers become necessary when the task has explicit structure to exploit) | ~2 hr |
| `PROFILE` | Latency breakdown of the 3 neural submodules only (causal/symbolic/actor), no training | seconds |
| `PROFILE2` | Full `get_action()` latency breakdown (featurize / host-device / submodule-forward / device-host / postproc) — the figure reported in the paper | ~1 min |
| `PROFILE3` | Cross-agent latency comparison (CCNSA vs CQL vs IQL) | ~1 min |
| `PROFILE4` | Peak GPU memory (inference-only and one training step) + CPU-only latency | ~1 min |

```bash
python ccnsa_revised.py --track A --seeds 20     # more seeds than the paper's default 10
python ccnsa_revised.py --track B                # Track B only
python ccnsa_revised.py --track RULES --quick     # 2-seed smoke test of the rule sweep
```

Run any long track under `nohup`/`tmux`/`screen` — several take hours:

```bash
nohup python ccnsa_revised.py --track A > run_track_a.log 2>&1 &
disown
tail -f run_track_a.log
```

Set `CCNSA_OUT=/path/to/output/dir` to control where results/checkpoints/JSON land (defaults to `./ccnsa_outputs`). Each track writes its own JSON (`track_a_results.json`, `track_b_results.json`, `ablation_only_results.json`, `rule_count_sweep_results.json`, `causal_method_sweep_results.json`, `cascade_ablation_results.json`, `memory_and_cpu_profile.json`, `latency_profile*.json`).

## Figures

```bash
python gen_fig0.py     # -> figures/fig0_architecture.png (architecture diagram)
python gen_figs.py      # -> figures/fig{2,3,4,5,6,7,8,9}_*.png, fig_faithfulness_DI.png
```

`gen_figs.py` plots the specific numbers reported in the paper directly (not read from a JSON file) — regenerate it after a fresh run only if you intend to update those constants to your own reproduced values.

## Reproducibility

- Seeds are fixed and enumerated in `SEEDS` (`42, 123, 456, 789, 101112, 202020, 303030, 404040, 505050, 606060`); `--seeds N` takes the first `N` deterministically, or extends past 10 with the same deterministic rule for higher-powered runs.
- On CUDA, `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False` are set explicitly — a real non-determinism bug (cudnn algorithm autotuning flipping deletion-insertion sign between identical-seed runs) was found and fixed during development; see the comment above `DEVICE` in `ccnsa_revised.py` for the full story.
- `--fresh-only` drops the original development seed set and runs on a disjoint replication set, used in the paper to confirm results aren't an artifact of seed selection.
- The symbolic-rule-count sweep (`RULES` track) and causal-discovery-method sweep (`CAUSAL` track) both use the same paired-significance/Bonferroni-correction machinery as the main ablation, at the same seed count, for comparable rigor — not a smaller, less-powered check.

## Known simplifications and honestly-disclosed limitations

This is a research codebase with disclosed, not hidden, simplifications. In particular:

- CQL and IQL are run in the same online interact-store-update loop as SAC/PPO/DT for a like-for-like comparison in this environment, rather than under a strict offline-only protocol.
- The main ablation study (properly powered, `n=10` seeds, paired significance testing) does **not** find the causal graph or symbolic-logic layer individually or jointly statistically necessary for reward, faithfulness, or deletion-insertion fidelity — nor does the follow-up `CASCADE` variant designed specifically to stress those components. This is reported as a genuine negative result in the paper, not concealed.
- Only the default 8-rule symbolic layer (`n_rules=8`) uses hand-curated target power levels; `r=4`/`r=16` variants (used only in the `RULES` sweep) use evenly-spaced synthetic targets — see the `SymbolicRuleLayer` docstring and `N_RULES_DEFAULT_U` comment in `ccnsa_revised.py`.
- The trained RL policy is evaluated only on the simulated environment; the real UNSW-NB15 track (Track B) validates causal-graph discovery, audit-rule precision, and classifier-baseline context, not the RL policy's sequential decisions on real data.

The paper's full Limitations section (§8.4) is the authoritative list; this README summarizes it for repository visitors who haven't read the paper.

## Data

Real UNSW-NB15 traffic (Track B) is fetched automatically on first use from the HuggingFace mirror:
[`https://huggingface.co/datasets/Mouwiya/UNSW-NB15`](https://huggingface.co/datasets/Mouwiya/UNSW-NB15)

Original dataset: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Moustafa & Slay, 2015).

## Citation

See `CITATION.cff`, or:

```bibtex
@article{singh2026ccnsa,
  author  = {Singh, Uday Pratap and Kumari, Bersha},
  title   = {CCNSA: A Neuro-Symbolic Reinforcement Learning Agent for Network Trauma Recovery with Faithful Explanations},
  journal = {Under review},
  year    = {2026}
}
```

Update this entry once the paper is accepted and a DOI/volume/page is assigned.

## License

Apache License 2.0 — see `LICENSE`.
