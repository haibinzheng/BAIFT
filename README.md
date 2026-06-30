# BAIFT

This repository contains the code, sample data, evaluation scripts, and visualization artifacts for **BAIFT**, a behavior-aware framework for intersectional fairness testing of large language models.

The artifact is prepared for anonymous review. It intentionally avoids author names, affiliations, and personal contact information in the repository-facing documentation.

## Overview

BAIFT targets fairness violations that are difficult to expose with outcome-only or semantic-only checks. Given a task anchor, BAIFT generates paired or grouped prompts that preserve the task intent while varying sensitive attributes. It then compares the target LLM responses with behavior-aware metrics to detect explicit discriminatory instances and implicit behavioral disparities.

BAIFT focuses on four capabilities:

- **Intersectional case generation**: instantiate single sensitive attributes and selected intersectional attribute pairs.
- **Behavior-aware oracle**: evaluate response differences across helpfulness, informativeness, specificity, refusal behavior, and respect-related behavior.
- **Prioritized testing**: rank generated cases with rule-based, learned, and judge-based scoring signals.
- **Diagnostic analysis**: report vulnerable groups, affected behavioral dimensions, and intersectional amplification patterns.

## Framework

The figure below summarizes the BAIFT workflow implemented in this artifact.

![BAIFT framework](Vis/framework.png)

## Repository Layout

```text
BAIFT_Code/
|-- APIs/                       # API and local inference entry points
|-- BAIFT-main/                 # main BAIFT execution entry
|-- Baselines/                  # baseline generators: METAL, CAFFE, LLM-Roleplay
|-- Datasets/                   # generated fairness-test cases
|-- Metrics/                    # metric and scoring entry points
|-- Models/                     # model wrappers
|-- Results/                    # inference and evaluation outputs
|-- Vis/                        # figures and visualization artifacts
|-- data/                       # physical storage for generated cases and responses
|-- eval/
|   |-- generate/               # generation scripts for each method
|   |-- infer/                  # model-specific inference scripts
|   |-- score/                  # scoring scripts for each method
|   `-- scripts/                # shared inference and scoring implementations
|-- infer/                      # inference backends
|-- src/
|   |-- core/                   # shared constants and schemas
|   |-- generators/             # BAIFT and baseline generators
|   |-- metrics/                # Sim(c), Fv(c), IoU/AUC, behavior metrics
|   |-- models/                 # API/local model abstractions
|   `-- scoring/                # rule-based, learned, judge, and aggregation logic
|-- requirements.txt
`-- run.py
```

## Installation

```bash
pip install -r requirements.txt
```

For API-based inference, set the endpoint and key expected by your backend. The code supports OpenAI-compatible endpoints through environment variables or command-line arguments.

```bash
export API_BASE="https://your-api-endpoint"
export API_KEY="your-api-key"
```

For local inference, start the corresponding local serving backend first, such as vLLM, and then run the local inference scripts under `eval/infer/local/`.

## Quick Start

Generate BAIFT cases:

```bash
python BAIFT-main/run.py pipeline \
  --method baift \
  --num 100 \
  --seed 42 \
  --output ./Datasets/generate/baift/custom/baift_cases.jsonl
```

Run inference:

```bash
python eval/scripts/infer.py \
  --input ./Datasets/generate/baift/custom/baift_cases.jsonl \
  --mode api \
  --api-base "$API_BASE" \
  --model "$MODEL_NAME" \
  --output ./Results/eval/baift/$MODEL_NAME/responses.jsonl
```

Run scoring:

```bash
python eval/scripts/score.py \
  --input ./Results/eval/baift/$MODEL_NAME/responses.jsonl \
  --output ./Results/eval/baift/$MODEL_NAME/scores.jsonl \
  --tau 0.12
```

## Supported Methods

The artifact includes BAIFT and three baselines. All methods share the same inference and scoring pipeline.

| Method | Purpose |
| --- | --- |
| `baift` | Behavior-aware intersectional fairness testing. |
| `caffe` | Counterfactual prompt generation baseline. |
| `metal` | Metamorphic identity-perturbation baseline. |
| `llm_roleplay` | Role-based paired-prompt baseline adapted from role-playing fairness probes. |

Example generation commands:

```bash
bash eval/generate/baift.sh
bash eval/generate/caffe.sh
bash eval/generate/metal.sh
bash eval/generate/llm_roleplay.sh
```

## Main Metrics

BAIFT records both detection metrics and diagnostic metrics.

| Metric | Meaning |
| --- | --- |
| `Sim(c)` | Semantic similarity between paired responses. Higher values indicate stronger semantic consistency. |
| `Fv(c)` | Behavior-aware fairness violation score. Higher values indicate stronger behavioral disparity. |
| `#DIs` | Number of discriminatory instances detected by the method. |
| `#EDIs` | Number of explicit discriminatory instances. |
| `#IDIs` | Number of implicit discriminatory instances. |
| `IDI Ratio` | Fraction of implicit discriminatory instances among detected discriminatory instances. |
| `VG(g)` | Vulnerable-group score for group-level diagnosis. |
| `Amp(c)` | Intersectional amplification score. Positive values indicate stronger joint-attribute unfairness. |
| `IoU` / `AUC` | Coverage overlap and threshold-integrated coverage statistics. |

## Reproducing the Evaluation Workflow

The full evaluation can be run in three stages.

1. **Generate test cases**

```bash
bash eval/generate/baift.sh
bash eval/generate/caffe.sh
bash eval/generate/metal.sh
bash eval/generate/llm_roleplay.sh
```

2. **Run model inference**

API examples are stored under:

```text
eval/infer/api/
```

Local examples are stored under:

```text
eval/infer/local/
```

3. **Score responses**

```bash
bash eval/score/score_baift.sh
bash eval/score/score_caffe.sh
bash eval/score/score_metal.sh
bash eval/score/score_llm_roleplay.sh
```

The output files are written under `Results/eval/` or `data/eval/`, depending on the script entry point.

## Visualization Artifacts

The `Vis/` directory stores figures used for checking and presenting the experimental results.

```text
Vis/framework.png              # BAIFT workflow
Vis/CoverageIoU.pdf            # RQ1 coverage curve
Vis/heatmap_Sim.pdf            # RQ1 fine-tuning Sim(c)
Vis/heatmap_Fv.pdf             # RQ1 fine-tuning Fv(c)
Vis/Radar-DIs.pdf              # RQ3 #DIs radar plot
Vis/Radar-Fv.pdf               # RQ3 Fv(c) radar plot
Vis/DiagnosisHIS2R.pdf         # RQ4 behavior-dimension diagnosis
Vis/DiagnosisVG.pdf            # RQ4 vulnerable-group diagnosis
Vis/DiagnosisAmp.pdf           # RQ4 amplification diagnosis
```

## Anonymous Artifact Notes

Before packaging the artifact for anonymous review, check the following:

- Do not include `.git/`, because Git metadata may contain author names, emails, and remote repository URLs.
- Do not include `.DS_Store`, `__pycache__/`, `.pytest_cache/`, or other local cache files.
- Do not include API keys, endpoint-specific secrets, private model credentials, or local absolute paths.
- Keep model names, method names, and experimental settings consistent with the manuscript.

The source files in this repository use environment variables for API credentials and do not require hard-coded secrets.

