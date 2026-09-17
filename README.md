<div align="center">

# 🩻 Broken Evidence

### Medical Vision–Language Models Answer Through Broken Evidence

**An evidence-conditional audit of medical VLMs, from radiologist-built 2D probes to simulator-built 3D CT probes whose answers are computed, not annotated.**

[![Paper](https://img.shields.io/badge/📄_Paper-coming_soon-1f6feb?style=for-the-badge)](#-citation)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-broken--evidence-ffcc4d?style=for-the-badge)](https://huggingface.co/datasets/jhq0709/broken-evidence)
[![Scored probes](https://img.shields.io/badge/scored_probes-355%2C062-8250df?style=for-the-badge)](#-at-a-glance)
[![Models](https://img.shields.io/badge/model_configs-29-2da44e?style=for-the-badge)](#-leaderboards)

[![3D annotation cost](https://img.shields.io/badge/3D_annotation_cost-zero-ff5a5f?style=flat-square)](#-two-constructions-one-evaluation)
[![Label provenance](https://img.shields.io/badge/label_provenance-8%2C476%2F8%2C476_verified-0969da?style=flat-square)](#-two-constructions-one-evaluation)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](#-quickstart)
[![Code](https://img.shields.io/badge/code-MIT-blue?style=flat-square)](LICENSE)
[![Data](https://img.shields.io/badge/data-CC_BY--SA_4.0_%7C_CC_BY--NC--SA_4.0-lightgrey?style=flat-square)](#-data-and-licences)

<img src="assets/teaser.png" width="96%" alt="A fluent answer and a safe refusal look alike until the evidence is checked"/>

<sub><b>A fluent answer and a safe refusal look alike until the evidence is checked.</b> In 2D the evidence is broken by masking and a false premise; in 3D the question asks about a counterfactual growth whose answer is computed from geometry.</sub>

</div>

---

## ⚡ TL;DR

> **A safe clinical model must recognise when the evidence for an answer has failed.** We formalise this as an **evidence contract** and measure its violation, **silent failure**: a fluent, confident answer given through broken evidence.

- 🧑‍⚕️ **2D, built by four board-certified radiologists.** 300 cases, 2,556 probes, 16 vision models. Capability and safe behaviour come apart, silent failure rises with clinical risk tier for most models, and a construction-blind radiologist leads the best model by **14.1 composite points**.
- 🧮 **3D, built by computation.** Whether a lesion would touch the aorta after growing *g* mm follows from **one distance transform**. That gives **9,484 verifiable probes over 588 CT volumes at zero annotation cost**, each about a state that never occurred, so memorisation is excluded by construction.
- 🔬 **The failure is localised.** Models see the outlined lesion (up to **81.7%**), name the target (up to **60.0%**) and compare two numbers perfectly (**100%**), yet read distance at chance (**23.0–28.3%**, chance 25%), on CT and on a blank synthetic background alike. Their decision variable tracks the growth number in the question (AUROC up to **0.96**), not the volume (**0.46–0.53**).

---

## 📊 At a glance

<div align="center">

| 🧾 Scored probes | 🤖 Model configurations | 🧠 CT volumes | ✅ Verified 3D labels | 💸 3D annotation cost |
|:---:|:---:|:---:|:---:|:---:|
| **355,062** | **29** | **588** | **8,476 / 8,476** | **0** |

</div>

| | 🩻 2D suite | 🧊 Volumetric suite |
|---|---|---|
| **Construction** | annotation-backed, four board-certified radiologists | simulator-backed, one Euclidean distance transform per lesion |
| **Sources** | VQA-RAD, SLAKE, ROCO, chest radiographs (300 cases) | Medical Segmentation Decathlon liver, lung, pancreas, colon (588 volumes) |
| **Probes** | 2,556 five-option MCQ, 240 counterfactual triplets | 9,484 computed probes, 4,238 matched pairs with opposite answers |
| **Reference** | construction-blind radiologist | reference rule re-derived from stored provenance on every label |
| **Perturbations** | paraphrase, negation, specificity drop, knowledge-only, traps, ROI-masked, ROI-only, laterality flip | the same text operators, ROI arms, blur sweep, identification control, sub-tasks, metric-channel control, oracles |
| **Models** | GPT, Claude, Gemini, Qwen, Kimi, LLaVA-Med, HuatuoGPT-V, 2 text-only baselines | Qwen2.5-VL 3B–72B, Qwen3-VL, InternVL3, SmolVLM2, LLaVA-OV, Idefics3, Pixtral, M3D-LaMed ×2, Med3DVLM |

---

## 🧭 Two constructions, one evaluation

<div align="center">
<img src="assets/construction.png" width="96%" alt="Annotation-backed 2D construction and simulator-backed 3D construction"/>
</div>

```mermaid
flowchart LR
    A["🩻 2D probe bank<br/>radiologist-adjudicated gold"] --> C{{"📜 Evidence contract<br/>valid evidence + well-posed premise"}}
    B["🧊 3D probe bank<br/>gold = 1[g ≥ d], computed"] --> C
    C --> D["🔀 Paired interventions<br/>text · visual · modality-specific"]
    D --> E["🤖 Medical VLM"]
    E --> F["🎯 Reference scoring<br/>exact match, no judge model"]
    F --> G["✅ supported · 🛑 safe refusal · ❔ uncertainty · ⚠️ silent failure"]
    G --> H["📐 Capability · Safety · Grounding → risk-weighted composite"]
```

<div align="center">
<img src="assets/evaluation.png" width="96%" alt="Shared evidence-conditioned evaluation"/>
</div>

---

## 🔥 Headline findings

### 1 · Capability is not safety

<img align="right" src="assets/audit2d.png" width="50%" alt="2D audit summary"/>

- The best model, **Claude Opus 4.7**, reaches a composite score of **69.2**; the construction-blind radiologist reaches **83.3**.
- Silent-failure rates on traps span **22.0–62.8%** across the sixteen models, against **5.8%** for the radiologist.
- **Gemini 3 Flash** is vision-anchored (VGR **+41.1 pp**) yet unsafe (SFR **37.2%**); **GPT-5.4-nano** is prior-driven (VGR **−27.1 pp**) and unsafe (**51.7%**). Two failure regimes, separable only because grounding and safety are measured apart.
- The ordering is stable: under nine alternative harm weightings and grounding normalisations, Spearman ρ ≥ **0.97**, the top five are unchanged and the leader is the same.

<br clear="right"/>

### 2 · Thirteen volumetric models, one coin

- On the growth-matched subset every model lands at **48.7–51.0%**, every interval containing 50.
- The result is threshold-free: the decision variable log p(yes) − log p(no) ranks the computed label at AUROC **0.470–0.522**.
- The volume is not ignored. **InternVL3-14B** changes **92.0%** of its answers when shown the CT and stays at **50.1%**.

### 3 · The score reads the question, not the volume

<div align="center">
<img src="assets/decision_variable.png" width="92%" alt="Decision variable AUROCs and the scale ladder"/>
</div>

- Swap only the growth number in the question: the decision variable ranks the two answers correctly at AUROC up to **0.959**.
- Swap only the volume under an identical sentence: **0.46–0.53** for every model.
- Scale sharpens the reading of the number and leaves the label untouched. **Qwen2.5-VL 3B → 72B**: number AUROC **0.54 → 0.96**, label AUROC **0.50 → 0.50**.

### 4 · Perception works, metric estimation is missing

<div align="center">
<img src="assets/subtasks.png" width="92%" alt="Sub-task decomposition"/>
</div>

| Sub-task on the `identified` rendering | Qwen2.5-VL-7B | InternVL3-8B | Qwen3-VL-8B | Qwen2.5-VL-32B |
|---|:---:|:---:|:---:|:---:|
| 🔴 Which organ holds the red-outlined lesion | 67.7 | 41.0 | **81.7** | 76.0 |
| 🔵 Which structure is outlined in cyan | 34.3 | 31.3 | **60.0** | 45.0 |
| 📏 How many mm separate them (chance 25) | 25.7 | 26.3 | 23.0 | 28.3 |

### 5 · Two ways the evidence contract breaks in the volume

<div align="center">
<img src="assets/metric_channel.png" width="92%" alt="Metric-channel control and the clinical sentence"/>
</div>

- 📐 **A channel the models lack on any image.** The distance question on synthetic outlines over **uniform grey**, over **real CT**, and on the **real lesion** gives **23.7–28.3%** in all twelve cells. Each model keeps the same answer habit regardless of what the outlines enclose.
- 💬 **A clinical sentence that disables arithmetic they have.** *"Is 21.4 ≥ 18.16?"* is solved at **100%** by every model tested. The same two numbers inside the clinical question collapse every model to a near-constant answer.

### 6 · Classic instruments change meaning in 3D

<div align="center">
<img src="assets/four_arm.png" width="92%" alt="Four-arm grounding decomposition"/>
</div>

- 🧩 **Grounding inverts.** VGR reaches **+49.5 pp** in 2D and never exceeds **+0.8 pp** in 3D (**−8.3 to +0.8 pp**): removing the evidence region costs nothing, removing everything else costs the volume.
- 🎲 **The scoring rule decides safety.** Raw likelihood over option strings reports a weighted silent-failure rate of **0.0%** for every model on every trap family; content-free calibration reveals **43.7–96.4%**.
- 🔒 **No memorisation.** Asked which public collection a scan comes from, models score **0.0–5.0%** while the organ control scores **50.0–57.5%**.

<details>
<summary><b>🖼️ What the volumetric models are shown</b></summary>

<div align="center">
<img src="assets/renderings.png" width="92%" alt="Identification-control renderings"/>
</div>

`plain` takes three centre slices, `bestslice` re-points them at the structures, `overlay` adds outlines on the centre slices, and `identified` combines joint-visibility slices, outlines and a 10 mm bar drawn from each panel's own spacing.
</details>

---

## 🏆 Leaderboards

<details open>
<summary><b>🩻 2D suite</b> (%, VGR in percentage points; ↓ lower is safer)</summary>

| Model | Overall | Original | SFR ↓ | VGR | PR | NEG | SDR | LPA | **CS** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 🧑‍⚕️ *Radiologist (reference)* | *92.4* | *95.3* | *5.8* | *+4.5* | *92.7* | *90.7* | *91.5* | *93.6* | ***83.3*** |
| 🥇 Claude Opus 4.7 | **76.5** | **78.7** | **22.0** | +22.9 | 78.0 | **83.1** | **79.9** | 98.6 | **69.2** |
| 🥈 Gemini 3.1 Flash-Lite | 70.7 | 73.0 | 22.3 | **+49.5** | 75.7 | 63.1 | 70.7 | 98.9 | 66.0 |
| 🥉 Gemini 3 Flash | 71.9 | 77.0 | 37.2 | +41.1 | **80.7** | 75.6 | 76.2 | 98.9 | 63.7 |
| GPT-5.5 | 69.4 | 75.0 | 36.8 | +12.0 | 74.3 | 68.9 | 59.8 | **99.6** | 61.3 |
| GPT-5.4 | 65.2 | 68.0 | 28.8 | +15.6 | 68.7 | 56.4 | 59.8 | 98.6 | 57.9 |
| Claude Sonnet 4.6 | 61.2 | 63.3 | 41.3 | +18.2 | 66.3 | 60.9 | 56.7 | 97.9 | 52.9 |
| HuatuoGPT-V 7B | 55.5 | 55.3 | 31.5 | +28.1 | 59.7 | 26.2 | 51.2 | 91.9 | 50.4 |
| LLaVA-Med 7B | 44.8 | 43.3 | 43.0 | +28.6 | 50.0 | 15.6 | 50.6 | 75.6 | 44.7 |
| GPT-4o | 56.1 | 59.3 | 53.0 | +3.6 | 63.3 | 45.8 | 50.6 | 98.6 | 44.1 |
| GPT-5.4-mini | 54.7 | 55.3 | 49.7 | −8.9 | 57.0 | 44.9 | 47.0 | 98.2 | 42.9 |
| Claude Haiku 4.5 | 49.6 | 46.7 | 49.0 | −9.4 | 51.0 | 32.4 | 42.1 | 97.5 | 39.9 |
| Qwen3.5-397B-A17B | 46.8 | 53.0 | 54.3 | +24.0 | 53.0 | 33.3 | 42.7 | 92.2 | 39.9 |
| Kimi-K2.6 | 46.6 | 49.3 | 50.8 | −2.1 | 47.7 | 36.0 | 37.8 | 95.4 | 38.8 |
| Kimi-K2.5 | 47.3 | 49.7 | 60.0 | +10.4 | 54.3 | 36.4 | 41.5 | 95.1 | 38.2 |
| GPT-5.4-nano | 38.8 | 31.7 | 51.7 | −27.1 | 29.7 | 16.9 | 25.0 | 90.5 | 31.6 |
| Qwen3.5-9B | 39.4 | 43.3 | 62.8 | +12.5 | 48.0 | 28.4 | 32.3 | 84.5 | 31.0 |

</details>

<details>
<summary><b>🧊 Volumetric suite</b> (growth-matched subset, 1,368 probes, chance 50%)</summary>

| Model | Input | Gate (/140) | Accuracy [95% CI] | Modal answer share | Pairs answered identically |
|---|---|---:|---|---:|---:|
| SmolVLM2-2.2B | montage | 121 | 50.0 [48.2, 51.8] | 100.0 | 99.8 |
| Qwen2.5-VL-3B | montage | 140 | 50.0 [48.2, 51.8] | 99.6 | 99.4 |
| Qwen2.5-VL-7B | montage | 140 | 50.0 [48.2, 51.8] | 100.0 | 100.0 |
| Qwen2.5-VL-32B | montage | 140 | 50.6 [48.6, 52.6] | 51.5 | 86.7 |
| Qwen3-VL-8B | montage | 140 | 48.7 [46.6, 50.7] | 79.5 | 88.8 |
| InternVL3-8B | montage | 140 | 50.1 [48.0, 52.2] | 72.6 | 79.4 |
| InternVL3-14B | montage | 140 | 50.1 [48.4, 51.9] | 96.5 | 94.1 |
| LLaVA-OneVision-7B | montage | 140 | 49.5 [47.7, 51.3] | 99.0 | 99.8 |
| Idefics3-8B | montage | 82 | 50.0 [48.2, 51.8] | 100.0 | 100.0 |
| Pixtral-12B | montage | 140 | 51.0 [48.8, 53.3] | 61.3 | 86.6 |
| M3D-LaMed-Phi3-4B | native 3D | 107 | 49.9 [48.1, 51.7] | 80.1 | 93.9 |
| M3D-LaMed-Llama2-7B | native 3D | 80 | 50.1 [48.2, 52.0] | 80.5 | 93.7 |
| Med3DVLM-7B | native 3D | 80 | 50.0 [48.2, 51.8] | 100.0 | 100.0 |

*Gate*: seven known-answer questions (*"Is this a CT scan?"*) on twenty volumes through the identical scoring path.

</details>

---

## 🗂️ Repository layout

```
broken-evidence/
├── 🩻 suite2d/                    annotation-backed 2D suite
│   ├── scripts/                   API + local runners, image perturbations, blur sweep, aggregation, validator
│   └── analysis/                  composite score, sensitivity, bootstrap CIs, clinician baseline, blur aggregation
├── 🧊 suite3d/                    simulator-backed volumetric suite
│   ├── spatialgen/                probe generation, rendering, runners, controls, oracles, sub-tasks
│   ├── analysis/                  identification control, text perturbations, metric channel, headline
│   ├── tools/                     render pre-cache, VRAM-aware GPU job queue
│   ├── reader_study/              104-probe reader form and answer key
│   └── growth_matched.py, analyse_*.py, make_figure_data.py
├── 🎨 figures/                    one script per paper figure (+ volumetric table)
├── 📈 results/                    2D aggregates and volumetric figure data behind every number
├── 🧰 scripts/fetch_data.py       pulls the dataset from the Hub and wires it into the tree
└── 🖼️ assets/                     README figures
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/hq0709/broken-evidence.git && cd broken-evidence
pip install -r requirements.txt
python scripts/fetch_data.py             # ~1.7 GB: probes, segmentations, every cached volumetric output
```

**Reproduce the volumetric headline in seconds, no GPU:**

```bash
python suite3d/growth_matched.py                  # 13-model table on the growth-matched subset
python suite3d/analysis/metric_control.py         # distance on grey / CT / real backgrounds
python suite3d/analyse_margin.py                  # how often the volume changes the answer
python figures/build_figs.py                      # paper figures -> figures/out/
```

**Audit your own model:**

```bash
# 2D: any OpenAI-compatible vision API (keys in .env)
python suite2d/scripts/bench_run_baseline.py --model gpt-4o --format mcq
python suite2d/scripts/bench_run_baseline.py --model gpt-4o --format mcq --no-image     # blind control

# 3D: register a Hugging Face model tag in suite3d/spatialgen/run_multimodel.py, then
export MSD_ROOT=/path/to/MSD            # Medical Segmentation Decathlon volumes
python suite3d/spatialgen/sanity_controls.py --model <hf-id> --data-root $MSD_ROOT --out gate.jsonl   # response-channel gate first
python suite3d/spatialgen/run_identification_control.py \
    --qa suite3d/cfqa_Task03_Liver/qa --task-dir $MSD_ROOT/Task03_Liver \
    --seg-cache suite3d/cfqa_Task03_Liver/seg_cache \
    --model <tag> --condition identified --subset matched --out my_model_liver.jsonl
```

---

## 🔁 Reproduce every number

| Paper result | Script | Needs |
|---|---|---|
| 2D headline table and composite | `suite2d/analysis/recompute_cscore.py` | `results/2d` |
| Composite robustness (9 variants) | `suite2d/analysis/cscore_sensitivity.py` | `results/2d` |
| 2D audit summary figure | `figures/build_fig_audit2d.py` | `results/2d` |
| 2D visual-information decay | `figures/build_figs.py` | `results/2d` |
| 2D metrics and bootstrap intervals from fresh runs | `suite2d/scripts/bench_aggregate.py`, `suite2d/analysis/bootstrap_cscore.py` | harness outputs |
| 2D blur-sweep table | `suite2d/analysis/blur_aggregate.py` | harness outputs |
| Volumetric headline table | `suite3d/growth_matched.py` | dataset |
| Volumetric table under the 2D columns | `suite3d/analysis/text_perturbations.py`, `suite3d/analysis/headline_3d.py`, `figures/build_tab_headline3d.py` | dataset |
| Identification control and ceilings | `suite3d/analysis/identification_control_ci.py` | dataset |
| Sub-task decomposition figure | `figures/build_figs.py` | dataset |
| Metric-channel control | `suite3d/analysis/metric_control.py`, `figures/build_fig_metric.py` | dataset |
| Decision variable and scale ladder | `figures/compute_decision_variable.py`, `figures/build_figs.py` | dataset |
| Answer changes under the volume | `suite3d/analyse_margin.py` | dataset |
| Four-arm grounding decomposition | `suite3d/make_figure_data.py`, `figures/build_figs.py` | dataset |
| Volumetric silent failure, raw against calibrated, with image contribution | `suite3d/analyse_traps.py` | dataset |
| Scoring rule: raw and content-free calibrated family metrics | `cd suite3d && python spatialgen/score_families.py --corpus families/all.jsonl --pred fam_qwen7b_sighted.jsonl --calib calib_qwen7b_sighted.jsonl` | dataset |
| Scored-probe total | `suite3d/analysis/count_scored_probes.py` | dataset |
| What the models are shown | `figures/build_fig_render.py` | dataset + MSD |

Every volumetric number is recomputed from per-probe outputs with volume-clustered bootstrap intervals.

---

## 📦 Data and licences

The dataset lives at **[🤗 jhq0709/broken-evidence](https://huggingface.co/datasets/jhq0709/broken-evidence)**.

| Part | Contents | Licence |
|---|---|---|
| `2d/` | manifest, 2,556 MCQ and open probes, triplets, ROI grounding, clinician baseline, images and perturbed images | CC BY-NC-SA 4.0 (source images keep their upstream terms; credentialed chest radiographs are obtained under their data use agreements) |
| `3d/` | 9,484 probes with provenance, TotalSegmentator masks, text perturbations, synthetic metric-control renders, cached model outputs for every reported experiment | CC BY-SA 4.0 (derived from the Medical Segmentation Decathlon) |
| code | this repository | MIT |

---

## 📝 Citation

```bibtex
@article{jiang2026brokenevidence,
  title   = {Medical Vision--Language Models Answer Through Broken Evidence},
  author  = {Jiang, Hanqi and Chen, Junhao and Kang, Mingyu and Kwon, Hyeokjae and Chen, Lifeng and
             You, Weihang and Pan, Yi and Gong, Haozhen and Ren, Hui and Li, Quanzheng and
             Liu, Tianming and Li, Xiang},
  year    = {2026},
  note    = {Under review}
}
```

## 🙏 Acknowledgements

Built on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/), [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), VQA-RAD, SLAKE, ROCO, MIMIC-CXR and CheXpert. We thank the four board-certified radiologists who built and adjudicated the 2D suite.

<div align="center">
<sub>University of Georgia · Harvard Medical School · Chungbuk National University · Chungnam National University Hospital · National University of Singapore</sub>
</div>
