# Prisoner Dilemma (Negotiation) Metrics — Report

**Generated:** 20260218_145322  
**Data source:** Deal-or-No-Dialog (DoND), split `validation`  
**Sample count:** 1087  
**Data note:** Each run loads the DoND JSONL from disk and runs the pipeline from scratch (no cached past test results). Metrics use the local preference model + Pareto only (no LLM); see sections 3.4–3.5 below.

## 1. Metric definitions (required matrix)

| Matrix Metric   | Source Field                                                   | Description                                      |
| --------------- | -------------------------------------------------------------- | ------------------------------------------------ |
| Accuracy        | `ai_assisted.advice_quality`                                   | Quality of AI coaching advice (0–1)              |
| Processing Time | `ai_assisted.ai_processing_time`                               | Time for AI to generate response (seconds)       |
| Success Rate    | `ai_assisted.success`                                          | Whether negotiation reached agreement (binary)   |
| Utility         | `ai_assisted.utility`                                          | Achieved utility score for the AI-assisted party |
| Satisfaction    | Derived (likely `human_only.satisfaction` or a combined score) | Negotiation satisfaction rating (1–5 scale)       |
| Improvement     | `(ai_utility - human_utility) / human_utility`                 | Relative gain over human-only baseline           |

## 2. Requirement checklist (DoND-based metrics)

| Matrix Metric   | Required Source / Description                                  | DoND-based implementation                                                        | Satisfied |
| --------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------- |
| Accuracy        | `ai_assisted.advice_quality` — Quality of AI coaching (0–1)   | Preference-estimation quality (cosine sim. pred vs. true prefs)                   | Yes       |
| Processing Time | `ai_assisted.ai_processing_time` — Time for AI response (s)    | Wall-clock for `estimate_preferences` + `best_offer`                              | Yes       |
| Success Rate    | `ai_assisted.success` — Agreement reached (binary)             | From dataset: deal present in `output`                                           | Yes       |
| Utility         | `ai_assisted.utility` — Achieved utility for party             | From dataset: utility of `my_final` under `my_values`                            | Yes       |
| Satisfaction    | Derived (human/combined) — Rating 1–5                          | **Not in DoND** — column present, value `null`                                   | No        |
| Improvement     | `(ai_utility - human_utility) / human_utility`                  | `(utility_if_followed_AI - achieved_utility) / achieved_utility`                  | Yes       |

**Summary:** 5 of 6 requirements are satisfied for DoND. Satisfaction is not available (DoND has no satisfaction ratings); the column is output as null.

### 2.1 Field descriptions (metrics matrix columns)

| Field | Description | Values / when NA |
| ----- | ----------- | ----------------- |
| **sample_id** | Unique identifier for the negotiation sample (e.g. dond_0001). | — |
| **scenario_id** | Scenario or dialogue ID from the dataset (if any). | Often empty for DoND. |
| **accuracy** | Preference-estimation quality: cosine similarity between predicted and true preferences (0–1). | **NA** when the model did not return a prediction. |
| **processing_time** | Time in seconds for the AI pipeline (estimate_preferences + best_offer) on this dialogue. | — |
| **success** / **success_display** | Whether the negotiation ended in a deal in the data. | Numeric: 1.0 = deal, 0.0 = no deal. Display: "Deal" / "No deal". |
| **utility** / **utility_display** | Achieved utility for the focal party (from the actual allocation in the data). | **NA** when no deal was reached. |
| **satisfaction** / **satisfaction_display** | Negotiation satisfaction rating (1–5). | **NA** for DoND (dataset has no satisfaction ratings). |
| **improvement** / **improvement_display** | Relative gain if the party had followed the AI suggestion vs. actual outcome: (AI_utility − achieved) / achieved. | **NA** when no deal (no achieved utility to compare) or when the AI produced no suggestion. |
| **rescuable** | Whether this *failed* negotiation could be "saved" by the system (AI suggested a Pareto deal). | **NA** when a deal was reached (not applicable). **Yes** when no deal and AI suggested a deal. **No** when no deal and AI had no suggestion. |
| **utility_if_saved** / **utility_if_saved_display** | For failed-but-rescuable cases: utility the focal party would get if they had followed the AI suggestion. | **NA** when not applicable (deal reached) or when not rescuable. |

## 3. Publication readiness (brief)

- **DoND-based metrics** are suitable for a journal if framed as **offline/counterfactual evaluation** of an AI negotiation coach on a real benchmark. Report mean ± std, n, and 95% CIs.
- **Simulated e2e** metrics are not suitable as primary publication evidence (synthetic outcomes).
- Do not claim in-the-wild effectiveness without a human study; frame as "offline evaluation" and "potential impact."
- Recommendations: use only DoND for main results; add baselines (e.g. no-AI); document reproducibility; state limitations (counterfactual improvement, single domain, no satisfaction in DoND).

### 3.1 Interpretation of improvement

- **Improvement** = (utility if party had followed AI suggestion − achieved utility) / achieved utility (only for samples where a deal was reached).
- **Negative improvement** means: the utility the party would get from the AI’s suggested deal is *lower* than the utility they actually achieved. So the human negotiators did *better* than what our AI would have suggested (e.g. the AI suggests a more balanced split; the human got a better share). That is expected in offline evaluation and is useful to report: the system is not yet beating observed human outcomes on average.
- **Positive improvement** means the AI suggestion would have been better than the actual deal for that party.
- This is a **counterfactual** comparison (we did not observe anyone following the AI); it is not a claim that the AI helps or hurts real users in the wild.

### 3.2 Failed negotiations that could be saved by the system (RAG / AI advice)

Separately, we can identify **negotiations that failed** (no deal in the data) but where our system (preference estimation + Pareto / RAG-backed advice) **would have suggested a valid deal**. Those are cases where the AI could in principle “save” a failed negotiation by proposing a Pareto-efficient split. The report below counts how many failed negotiations are **rescuable** (AI produced a suggestion) and, for those, the mean utility the focal party would get if they had followed that suggestion.

### 3.3 Data integrity (no fake data or hallucinations)

- **DoND report:** All metrics are computed from the loaded dataset and the real pipeline. Success and utility come from parsed dataset fields. Processing time is wall-clock. Accuracy is reported only when the model returns a prediction (no ground-truth substitution when the model fails). Improvement and rescuable use the same pipeline; when the model fails, the suggestion uses equal weights only (no GT).
- **E2E (simulated):** When using `--source e2e`, results come from the end-to-end test's simulated outcomes (parametric + random); they are not real negotiation data and are labeled as simulated.

### 3.4 Model / runtime (no local LLM in metrics)

- **DoND metrics do not use a local LLM** (e.g. Ollama). The pipeline uses:
  - **Preference estimation:** a local **DistilBERT-based** model (Hugging Face `transformers`), run on CPU/GPU. No API or Ollama.
  - **Pareto suggestion:** deterministic **best_offer** (pure Python) from `app.pareto`.
- So the test runs fully **offline** with no LLM server. The **coach/advice UI** (Gradio) can use Ollama or Gemini for text advice; that path is not used when computing `--source dond` metrics.

### 3.5 Validity of metrics without LLM — and how the platform helps

**Are the test/metrics valid without the LLM?** Yes. In the full app (`get_advice`), the **content** of the advice (what each party values, what allocation to suggest) comes from the same two pieces we evaluate:

1. **Preference estimation** — infers weights from dialogue (the model we measure with *accuracy*).
2. **Pareto suggestion** — `best_offer()` produces a concrete allocation (what we use for *improvement* and *rescuable*).

The LLM is then given a **hint** derived from that suggestion and turns it into natural-language advice. So we are evaluating the **engine** that decides *what* to recommend; the LLM only **phrases** it. Metrics on preference quality and Pareto suggestions are therefore valid for the system’s decision quality.

**How does the platform help negotiation without the LLM?** The platform can help even when the LLM is off or unavailable:

- **From dialogue:** The preference model infers what each side cares about (e.g. “you” values books more than hats). That understanding can be shown as structured hints or simple text (e.g. “Your inferred priorities: books > balls > hats”).
- **From Pareto:** `best_offer()` outputs a specific split (e.g. “2 books, 0 hats, 1 ball for you”). The UI can display that as a **suggestion** or **target allocation** without any LLM (e.g. “Suggested split: …” or “Consider offering: …”).
- **Optional LLM:** The LLM then adds fluent, contextual prose (e.g. “Given their focus on the trilogy, offering them the hats in exchange for …”). That improves readability and framing but is not required for the core recommendation.

So the metrics are valid because they measure the same preference + Pareto pipeline that drives advice content; and the platform helps negotiation by inferring preferences and suggesting Pareto-efficient deals, with or without the LLM.


---

## 4. Results

### 4.1 Summary statistics

| Metric | Mean | Std | 95% CI lower | 95% CI upper | Count |
| ------ | ---- | --- | ------------ | ------------ | ----- |
| accuracy | 0.7699 | 0.0868 | 0.7648 | 0.7751 | 1087 |
| processing_time | 0.0138 | 0.0027 | 0.0137 | 0.0140 | 1087 |
| success | 0.7764 | 0.4166 | 0.7517 | 0.8012 | 1087 |
| utility | 1.2750 | 0.4840 | 1.2423 | 1.3076 | 844 |
| satisfaction | — | — | — | — | 0 |
| improvement | -0.4323 | 0.4899 | -0.4654 | -0.3993 | 844 |

### 4.2 Failed negotiations that could be saved by the system (RAG / AI advice)

| Stat | Value |
| ---- | ----- |
| Negotiations that **failed** (no deal in data) | 243 |
| Of those, **rescuable** (AI suggested a Pareto deal) | 243 |
| Rescuable as % of failed | 100.0% |
| Rescuable as % of all | 22.4% |
| Mean utility for focal party if they had followed AI (rescuable only) | 0.6164 |

*Interpretation:* Among negotiations that ended with no deal, the system (preference estimation + Pareto / RAG-backed advice) could still propose a valid allocation in 243 cases. Those represent opportunities where the AI could in principle "save" a failed negotiation.


### 4.3 Metrics matrix (sample of rows; descriptive values)

| sample_id | scenario_id | accuracy_display | processing_time_display | success_display | utility_display | satisfaction_display | improvement_display | rescuable | utility_if_saved_display |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dond_0000 |  | 0.8278 | 0.0356 | No deal | NA | NA | NA | Yes | 0.5000 |
| dond_0001 |  | 0.7961 | 0.0174 | Deal | 1.7500 | NA | -0.8571 | NA | NA |
| dond_0002 |  | 0.7731 | 0.0159 | Deal | 1.2500 | NA | -0.6000 | NA | NA |
| dond_0003 |  | 0.7385 | 0.0152 | Deal | 0.7143 | NA | 0.2000 | NA | NA |
| dond_0004 |  | 0.7339 | 0.0154 | Deal | 0.7143 | NA | 0.2000 | NA | NA |
| dond_0005 |  | 0.8391 | 0.0120 | No deal | NA | NA | NA | Yes | 0.3333 |
| dond_0006 |  | 0.8059 | 0.0112 | No deal | NA | NA | NA | Yes | 1.0000 |
| dond_0007 |  | 0.8436 | 0.0110 | Deal | 1.6000 | NA | -0.5000 | NA | NA |
| dond_0008 |  | 0.7388 | 0.0104 | Deal | 2.0000 | NA | -1.0000 | NA | NA |
| dond_0009 |  | 0.6864 | 0.0149 | Deal | 2.0000 | NA | 0.0000 | NA | NA |
| dond_0010 |  | 0.5360 | 0.0149 | Deal | 1.1429 | NA | -0.7500 | NA | NA |
| dond_0011 |  | 0.7734 | 0.0155 | Deal | 1.0000 | NA | -1.0000 | NA | NA |
| dond_0012 |  | 0.8687 | 0.0151 | Deal | 2.0000 | NA | -0.5000 | NA | NA |
| dond_0013 |  | 0.7468 | 0.0160 | Deal | 0.8333 | NA | 0.2000 | NA | NA |
| dond_0014 |  | 0.7912 | 0.0170 | Deal | 1.2000 | NA | 0.3333 | NA | NA |
| dond_0015 |  | 0.8579 | 0.0112 | Deal | 0.7500 | NA | -0.8333 | NA | NA |
| dond_0016 |  | 0.8980 | 0.0109 | Deal | 1.0000 | NA | -0.6667 | NA | NA |
| dond_0017 |  | 0.8502 | 0.0179 | Deal | 1.7500 | NA | -0.4286 | NA | NA |
| dond_0018 |  | 0.7344 | 0.0175 | Deal | 2.2500 | NA | -1.0000 | NA | NA |
| dond_0019 |  | 0.6837 | 0.0107 | Deal | 1.0000 | NA | -0.4286 | NA | NA |
| dond_0020 |  | 0.7693 | 0.0112 | Deal | 0.9000 | NA | 0.0000 | NA | NA |
| dond_0021 |  | 0.7632 | 0.0114 | No deal | NA | NA | NA | Yes | 0.7000 |
| dond_0022 |  | 0.7235 | 0.0102 | No deal | NA | NA | NA | Yes | 0.5000 |
| dond_0023 |  | 0.7950 | 0.0124 | Deal | 2.0000 | NA | 0.0000 | NA | NA |
| dond_0024 |  | 0.6783 | 0.0114 | Deal | 2.0000 | NA | -0.5000 | NA | NA |

*... and 1062 more rows.*

---

## 5. Reproducibility

**Increasing the number of samples:** Omit `--max-samples` to use the **full split** (validation ≈1088, train ≈10095). Or set e.g. `--max-samples 500`. Use `--split train` for the larger training set.

From project root:

```bash
python experiments/data_processing/prisoner_dilemma_metrics.py --source dond --project prisonerdilemma_dond --split validation
```

To regenerate this single report file (analysis + results):

```bash
python experiments/data_processing/prisoner_dilemma_metrics.py --report --split validation
```
