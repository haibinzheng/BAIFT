# BAIFT

**A Behavior-Aware Framework for Intersectional Fairness Testing of LLMs**

Research artifact for the paper accepted at **ICSE 2027**.

**Haibin Zheng, Gan Xu, Jinyin Chen, and Zhenyang Yu** — Zhejiang University of Technology.

[Artifact and reproducibility guide](docs/ARTIFACT.md) · [Citation](CITATION.cff) · [Apache 2.0 license](LICENSE)

BAIFT investigates differences in the assistance language models provide to counterfactual users. The paper combines behavioral scoring along helpfulness, informativeness, supportiveness, respectfulness, and refusal strength with intersectional testing, prioritization, and diagnostics.

Behavioral disparity is a testing signal. Whether a difference is unfair depends on the task and context. Reasonable accommodations, useful group-specific assistance, and necessary safety restrictions require contextual judgment. The response-level judge does not decide discrimination or whether refusal is justified.

![BAIFT framework](Vis/framework.png)

## Artifact status

This repository contains source modules, small paired-case examples, inference/scoring entry points, figures, and the submitted supplementary material. **The current checkout is not a complete one-command reproduction of the paper.**

The current generation entry point replays files under `data/formal_datasets/`, which are not included; it does not implement the paper's complete search algorithm. Full behavioral scoring requires an externally supplied trained five-dimensional BERT-family checkpoint. See the [implementation audit](docs/ARTIFACT.md) for precise availability and limitations.

The existing examples are not the full experimental corpus or independent human-validation study. No private per-example experiment records are added in this repository preparation. The supplementary PDF is the submitted version, pending the finalized camera-ready replacement.

## Quick inspection

```bash
git clone https://github.com/haibinzheng/BAIFT.git
cd BAIFT
python scripts/check_artifact.py
```

This offline check uses only the Python standard library, validates source syntax, documentation links and example schemas, and reports missing reproduction inputs without printing example content. It downloads no weights and makes no API calls. Passing it does not establish paper-result reproduction.

Use `python scripts/check_artifact.py --require-full` to require the declared full-reproduction inputs as well. That check is expected to fail while formal datasets and a trained scorer checkpoint are unavailable.

## Model-dependent workflow

The source uses Python 3.10+ syntax. The existing dependency pins have been retained; a clean model-dependent environment installation remains to be recorded.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell use `.\.venv\Scripts\Activate.ps1`. Local serving dependencies may require Linux/CUDA. The offline check does not need the model packages.

Run from the repository root. First convert the supplied packed examples to the flat schema required by inference. With a compatible local model:

```bash
mkdir -p outputs
python scripts/prepare_pairs.py \
  --input data/generate/baift/llama-3.1-8b/samples.jsonl \
  --output outputs/cases.jsonl
python eval/scripts/infer.py \
  --input outputs/cases.jsonl \
  --mode local --model-path /path/to/compatible/model \
  --output outputs/responses.jsonl
```

Inference may consume substantial compute or incur API costs. The current API wrapper does not attach an authentication header; use a compatible local endpoint or adapt it before calling an authenticated provider. Its current defaults are temperature 0.7 and 1024 tokens, which do not establish the historical experiment configuration.

With an actual trained five-dimensional checkpoint and configured judge:

```bash
python eval/scripts/score.py \
  --input outputs/responses.jsonl --output outputs/scores.jsonl \
  --learned-model /path/to/trained-checkpoint --judge
```

The judge model defaults to `gpt-4o-mini`; supply its credentials through `BAIFT_JUDGE_API_KEY` or `OPENAI_API_KEY`. Never commit credentials. Add `--tau` for counts only with a documented calibrated threshold. Code/script defaults are not verified formal-experiment settings.

**Inspect failure messages.** The scoring script currently sets `sim=1.0` if semantic scoring fails; the judge can use heuristic fallback when disabled or on failure. Those runs are not equivalent to successful configured semantic/judge evaluation.

## Repository map

| Path | Contents |
| --- | --- |
| `src/generators/` | Current method-specific dataset replay adapters |
| `src/scoring/` | Rule, learned, judge, and hybrid behavioral scoring |
| `src/metrics/` | Pair classification, disparity, calibration and coverage utilities |
| `src/core/` | Schemas and default constants |
| `infer/`, `eval/scripts/` | Target inference and scoring entry points |
| `eval/generate/`, `eval/infer/`, `eval/score/` | Shell wrappers; check paths and settings before use |
| `data/` | Existing small prompt/response examples |
| `Vis/` | Framework/result figures and representative-case HTML |
| `docs/`, `scripts/` | Artifact guide and offline checkout check |
| `SupplementaryMaterials-BAIFT.pdf` | Submitted supplement |

The `.py` compatibility entries in `APIs/`, `BAIFT-main/`, `Baselines/` and `utils/` now contain Python wrappers. Other legacy directory aliases can still be plain path files; prefer the root entry point and `src/`.

## Interpreting results

Algorithm-flagged EDIs satisfy `Sim(c) < 0.9`; IDIs satisfy `Sim(c) >= 0.9` and `Fv(c) > tau`. DI is their total. These operational categories are distinct from human-confirmed unfairness. IDI/DI is not a detection rate among all evaluated pairs.

`Amp(c)` is joint disparity exceeding the larger corresponding single-attribute disparity; it does not isolate a matched interaction effect.

**Legacy output keys:** `paper_metrics()['DIs']` contains the EDI count, `IDIs` the IDI count, and `DIS` their total. This preparation retains these historical keys and does not alter experimental results.

## Citation and contact

[CITATION.cff](CITATION.cff) provides the supplied author order and title. Proceedings metadata and DOI will be added when supplied by the publisher. No artifact badge is claimed.

| Author | Affiliation | Email |
| --- | --- | --- |
| Haibin Zheng | Zhejiang University of Technology | haibinzheng320@gmail.com |
| Gan Xu | Zhejiang University of Technology | xu942073103@gmail.com |
| Jinyin Chen | Zhejiang University of Technology | chenjinyin@zjut.edu.cn |
| Zhenyang Yu | Zhejiang University of Technology | fishocean13@gmail.com |

The existing [Apache License 2.0](LICENSE) is retained. External datasets and models retain their own terms. For reproducibility questions, open a GitHub Issue with versions and a minimal example, omitting credentials and private responses.
