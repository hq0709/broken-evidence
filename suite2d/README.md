# 🩻 2D suite — annotation-backed evidence probes

300 cases from VQA-RAD, SLAKE, ROCO and chest-radiograph collections, built in three stages by four
board-certified radiologists into **2,556 five-option probes**: originals, paraphrase, negation, specificity
drop, knowledge-only rewrites, hallucination traps, ROI-masked, ROI-only and laterality-flipped variants.

## Run a model

```bash
python scripts/fetch_data.py --only 2d                        # from the repository root
cd suite2d
python scripts/bench_run_baseline.py --model gpt-4o --format mcq            # OpenAI-compatible APIs, Claude, Gemini
python scripts/bench_run_baseline.py --model gpt-4o --format mcq --no-image # matched blind control
python scripts/bench_run_huatuo.py   --format mcq                           # HuatuoGPT-Vision-7B: weights in suite2d/checkpoints/, official repo on PYTHONPATH
python scripts/bench_run_llava_med.py --format mcq                          # LLaVA-Med v1.5 (official package)
python scripts/bench_aggregate.py                                           # per-model metrics
python analysis/bootstrap_cscore.py                                         # bootstrap intervals on every axis
```

API keys are read from the environment or a `.env` file at the repository root.

## Visual-information decay

```bash
python scripts/bench_blur_sweep_make.py                       # blurred renders, sigma in {0,2,4,8,16,32,64}
python scripts/bench_blur_sweep_run.py --model gpt-4o --sigma 16   # one model at one sigma (0 ... 64, inf)
python analysis/blur_aggregate.py                             # -> results/2d/blur_aggregate.csv
```

## Composite score and robustness

```bash
python analysis/recompute_cscore.py           # Capability, Safety (risk-weighted SFR), Grounding -> CS
python analysis/cscore_sensitivity.py         # nine alternative weightings and grounding forms
python analysis/clinician_baseline_cscore.py  # construction-blind radiologist reference
python scripts/bench_validate_full.py         # integrity checks on the released suite
```

## Metrics

| Axis | Definition |
|---|---|
| Capability | mean of original, paraphrase, negation and specificity-drop accuracy |
| Safety | 100 − risk-weighted silent-failure rate, tier weights (1, 2, 3, 5, 8) |
| Grounding | ½ · (clip(VGR + 50) + ROI-masked accuracy), VGR = ROI-only − ROI-masked accuracy |
| CS | harmonic mean of the three axes, so no axis can compensate for another |
