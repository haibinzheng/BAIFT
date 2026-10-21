# Artifact and reproducibility guide

This guide records the actual contents of the BAIFT checkout during preparation of the ICSE 2027 camera-ready artifact. Paper acceptance is not an artifact certification.

## Availability audit

| Component | Current state and limitation |
| --- | --- |
| Generation | Four adapters call a formal-dataset loader. The four `data/formal_datasets/*.jsonl` files are absent; `--num` and `--seed` are ignored by current adapters. This is replay infrastructure, not the complete search/prioritization implementation. |
| Examples | Existing `data/generate/` and `data/eval/` files illustrate paired-case interfaces; they are not the full study. |
| Target inference | API/Transformers backends are present. API wrapper lacks authorization headers. Current generation length defaults to 1024; the manuscript describes 512. Historical settings must be verified from experiment configuration. |
| Semantic scoring | `src/metrics/lsa.py` uses SentenceTransformer `all-MiniLM-L6-v2` cosine similarity. Despite the LSA name, it is embedding-based. Reconcile the experimental semantic scorer with the manuscript. |
| Learned scoring | Five-label Hugging Face loader is present; trained checkpoint, training configuration and evaluation summary are absent. |
| Judge | Scores one prompt/response on five dimensions, without comparing groups or judging discrimination/refusal justification; default model is `gpt-4o-mini`. Actual configuration and fallback occurrence need records. |
| Calibration | Utilities are present; formal calibration input, calibrated weights/threshold and split relationships need documentation. |
| Metrics | Historical `paper_metrics()` key `DIs` means EDI; `DIS` means total DI. Preserve this distinction when reading exports. |
| Coverage | `iou_auc.py` computes quantized behavioral-score set overlap. Its equivalence to the paper's stated case-coverage definition needs verification. |
| Figures/supplement | Figures and submitted supplement are included; complete figure-regeneration inputs/scripts and camera-ready supplement are not yet supplied. |

## Reproduction inputs

The current replay adapters expect `data/formal_datasets/baift.jsonl`, `caffe.jsonl`, `metal.jsonl`, and `llm_roleplay.jsonl`. These use method-specific schemas. Renaming the flat paired examples is not a valid substitute.

The learned checkpoint must expose exactly the five behavioral labels through `id2label`. Its loader uses sigmoid for `multi_label_classification`, otherwise softmax. The checkpoint configuration must match its actual training target. Supply the actual checkpoint through `--learned-model` or `BAIFT_LEARNED_MODEL`; a generic untrained BERT checkpoint is not equivalent.

The judge API branch requests temperature 0, top-p 1 and 160 tokens. Normal 1–10 integer ratings map using `(score-1)/9`; the implementation also accepts [0,1] scores. Configure the model/endpoint with `BAIFT_JUDGE_MODEL` and `BAIFT_JUDGE_BASE_URL`, and credentials through the documented environment variables. Those implementation defaults are distinct from verified formal settings.

Scoring currently catches semantic errors and assigns `sim=1.0`; judge errors/disabled mode can produce heuristic scores. Inspect logs before interpreting output. A validated evaluation should record scorer provenance or explicitly fail, rather than count fallback as equivalent evaluation.

## Experimental reporting

Record target/judge identifiers, access dates, decoding settings, environment, calibrated weights/threshold, query counts, actual evaluated pair counts and runtime for each run. Target-model generation and judge generation are different settings.

The clarified principal analysis uses five randomized repetitions and one-sided paired t-tests after run aggregation: ten model pairs for Table III and five attribute pairs for Table IV. Variation between models is not run-to-run uncertainty. SD/CI must use actual repeat summaries.

EDI and IDI are operational flags. Human-confirmation results require their own denominators. Reasonable assistance and necessary safety restrictions need task/context judgments. `Amp` measures excess joint disparity, not a matched attribute interaction.

The main comparison has ten models; Qwen3-8B is additionally used for fine-tuning, bringing the total to eleven. No new experimental results or reconstructed private records are added by this preparation.

## Work remaining for full reproduction

1. Identify or restore the real generation/search implementation and approved distributable inputs.
2. Provide the trained scorer or its access mechanism, training setup and split documentation.
3. Reconcile semantic scoring, coverage, target decoding and calibration with actual experiment records.
4. Supply approved five-run aggregates and scripts mapping them to paper tables/figures. Private per-example responses are not needed for run-level summaries.
5. Finalize human-label protocol, exact confirmation counts and sample-set relationships from records. Training is confirmed disjoint from the 1,600-pair evaluation; other overlaps must not be inferred.
6. Record a clean install on an actual target environment and archive a stable version after the camera-ready materials are finalized.
