# 🧊 Volumetric suite — simulator-backed evidence probes

*"If this lesion grew by g mm in every direction, would it contact the aorta?"* Contact after isotropic
growth holds if and only if the lesion-to-target surface gap is at most g, so **one Euclidean distance
transform per lesion answers every such question exactly**. Matched pairs straddle the gap, so labels are
balanced by construction and a model answering both members identically is not computing the geometry.

| | |
|---|---|
| Sources | Medical Segmentation Decathlon: liver 118, lung 63, pancreas 281, colon 125 volumes |
| Targets | 29 structures segmented with TotalSegmentator (3 mm mode) |
| Probes | 9,484 (8,476 growth-contact, 4,238 matched pairs), margin 2 mm, cap 40 mm |
| Verification | reference rule re-derived from stored provenance on 8,476 / 8,476 labels |

## Layout after `scripts/fetch_data.py`

| Path | Contents |
|---|---|
| `cfqa_<task>/qa/` | the full probe corpus with provenance (gap, growth, rule) |
| `cfqa_<task>/seg_cache/` | TotalSegmentator anatomy masks |
| `common_subset/qa/all.jsonl` | the 2,262-probe subset every model is audited on |
| `cfqa_text/` | paraphrase, negation and specificity-drop variants |
| `mm_<model>_{sighted,blind}.jsonl` | 13-model sighted and blind runs |
| `sanity_<model>.jsonl` | response-channel gate |
| `roi_<task>_<model>_<arm>[_local\|_air].jsonl` | four-arm grounding decomposition; the headline uses tissue (`local`) fill |
| `decay_<model>.jsonl` | blur sweep |
| `families/all.jsonl`, `families/traps_v4.jsonl` | five-family probe corpus and anatomy-verified trap corpus |
| `fam_ / calib_ / v4_ / v4cal_` | family and trap runs with their content-free calibration |
| `tv_ / tvcal_ / v3_ / v3cal_ / tp_` | additional trap-family and target-contrast runs (included in the scored-probe total) |
| `results_new/id_*` | identification control, sub-tasks, oracles, input richness |
| `results_new/metric_*`, `results_new/text_*`, `results_new/leak_*` | metric-channel control, text perturbations, pretraining-overlap probe |

## Generate, render, run

```bash
export MSD_ROOT=/path/to/MSD
python spatialgen/run_pipeline.py --input $MSD_ROOT/Task03_Liver/imagesTr --outdir cfqa_Task03_Liver --fast   # segmentation + probes
python tools/prerender.py --conditions plain identified        # cache renderings
python spatialgen/sanity_controls.py --model Qwen/Qwen2.5-VL-7B-Instruct --data-root $MSD_ROOT --out gate.jsonl
python spatialgen/run_multimodel.py --qa common_subset/qa/all.jsonl --data-root $MSD_ROOT --model Qwen/Qwen2.5-VL-7B-Instruct --out mm.jsonl
python spatialgen/run_identification_control.py --qa cfqa_Task03_Liver/qa --task-dir $MSD_ROOT/Task03_Liver \
    --seg-cache cfqa_Task03_Liver/seg_cache --model qwen7b --condition identified --subset matched --out id.jsonl
python spatialgen/run_subtasks.py --subtask distance --model qwen7b --out distance.jsonl
python spatialgen/run_metric_control.py --model qwen7b --background grey --out metric_grey.jsonl
```

## Analyse (CPU, from released outputs)

```bash
python growth_matched.py                        # headline table on the growth-matched subset
python analysis/identification_control_ci.py    # identification control and ceilings
python analysis/text_perturbations.py           # PR / NEG / SDR
python analysis/metric_control.py               # distance on grey / CT / real backgrounds
python analysis/headline_3d.py                  # the volumetric suite under the 2D columns
python analyse_margin.py                        # perturbation, decision gap, answer changes
python make_figure_data.py                      # figure data, including the four-arm decomposition
python analysis/count_scored_probes.py          # scored-probe total
```
