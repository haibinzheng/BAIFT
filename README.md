# BAIFT

This repository is a fairness evaluation pipeline centered on BAIFT.

## Introduction

BAIFT (Bias Attribution in Fairness Testing) is an intersectional fairness testing framework for large language models. The core idea is to construct paired prompts that keep the task intent as stable as possible while varying sensitive identity attributes, then measure whether the model produces behaviorally different responses under those controlled changes.

## BAIFT

BAIFT is the main method in this project.
In this repository, BAIFT is used to:

- generate paired fairness test cases
- run model inference on the paired prompts
- evaluate semantic consistency and fairness-related response disparity

## How to use

Install:

```bash
pip install -r requirements.txt
```

Generate BAIFT cases:

```bash
python BAIFT-main/run.py pipeline \
  --method baift \
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
  --output ./Results/eval/baift/$MODEL_NAME/lsa_scores.jsonl \
  --tau 0.12
```

## Baselines

Other supported methods:

- CAFFE: counterfactual prompts
- METAL: identity-prefix perturbation
- LLM-RolePlay: role-based paired prompts

They use the same inference and scoring pipeline as BAIFT.

## Project structure

The capitalized directories are the main user-facing entries. Some of them are
symbolic links to the implementation or data directories, so the same files are
not duplicated. Existing generated datasets, model responses, score files, and
placeholder directories are retained.

```text
BAIFT/
├── APIs/                              # inference backends (links to infer/)
│   ├── api.py                         # OpenAI-compatible API inference
│   └── local.py                       # local model inference
├── BAIFT-main/                        # main BAIFT entry (links to root code)
│   ├── run.py                         # generation pipeline CLI
│   ├── infer/
│   │   ├── api.py
│   │   ├── local.py
│   │   └── test_infer.py
│   └── src/
│       ├── core/                      # constants and shared schemas
│       ├── generators/                # BAIFT and baseline generators
│       ├── metrics/                   # LSA, FV, IoU/AUC, disparity, behavior
│       ├── models/                    # API/local model abstractions
│       └── scoring/                   # rule, learned, judge, aggregation
├── Baselines/                         # baseline generator entries
│   ├── caffe.py
│   ├── formal_datasets.py
│   ├── metal.py
│   └── roleplay.py
├── Datasets/                          # generated input datasets
│   └── generate/                      # link to data/generate/
│       ├── baift/<model>/samples.jsonl
│       ├── caffe/<model>/samples.jsonl
│       ├── llm_roleplay/<model>/samples.jsonl
│       └── metal/<model>/samples.jsonl
├── Metrics/                           # metric and scoring entries
│   ├── metrics/
│   │   ├── behavior.py
│   │   ├── disparity.py
│   │   ├── fv.py
│   │   ├── iou_auc.py
│   │   └── lsa.py
│   └── scoring/
│       ├── aggregator.py
│       ├── learned.py
│       ├── llm_judge.py
│       └── rule_based.py
├── Models/                            # model implementations
│   └── models/
│       ├── base.py
│       ├── api.py
│       └── local.py
├── Results/                           # retained inference/evaluation outputs
│   └── eval/                          # link to data/eval/
│       ├── baift/<model>/
│       │   ├── responses.jsonl       # model responses
│       │   └── *_scores.jsonl        # metric output, when generated
│       ├── caffe/<model>/
│       ├── llm_roleplay/<model>/
│       └── metal/<model>/
├── Vis/                               # visualization outputs
├── eval/                              # executable evaluation workflow
│   ├── generate/                      # one generation script per method
│   │   ├── baift.sh
│   │   ├── caffe.sh
│   │   ├── llm_roleplay.sh
│   │   └── metal.sh
│   ├── infer/                         # response generation per model
│   │   ├── api/                   # hosted/API model scripts
│   │   └── local/                 # local model scripts
│   ├── score/                         # one evaluation script per method
│   │   ├── score_baift.sh
│   │   ├── score_caffe.sh
│   │   ├── score_llm_roleplay.sh
│   │   └── score_metal.sh
│   ├── scripts/
│   │   ├── infer.py               # shared inference implementation
│   │   └── score.py               # shared evaluation implementation
│   └── run_vllm.sh                    # local vLLM server helper
├── data/                              # physical storage behind links above
│   ├── generate/                      # generated cases; keep result files
│   └── eval/                          # responses and scores; keep results
├── infer/                             # physical inference implementation
├── src/                               # physical core implementation
├── run.py                             # physical pipeline entry point
└── utils/
    └── utils.py
```