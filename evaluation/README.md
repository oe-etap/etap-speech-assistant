# Response evaluation

This package scores the LLM replies of one speech-to-speech pipeline run
(ASR → LLM → TTS) with methods taken from the literature. The input is the
`transcripts.yaml` or `transcripts.jsonl` that the run already wrote. The output
is a readable report, a CSV, a JSON file, and optionally BibTeX.

Working principle: **tier 0 is decidable; everything else is an estimate.** The
report names the source and the validation status of every method, so the
numbers remain traceable.

The package does three jobs: score one run; compare two runs pairwise when the
question is whether a parameter change improved the replies (see
[Comparing two runs](#comparing-two-runs-for-parameter-tuning)); and score and
compare a whole configuration grid in one pass (see
[Evaluating a run grid](#evaluating-a-run-grid-in-one-pass)).

The package is self-contained: it imports nothing from the rest of the pipeline,
and it can be called in two ways. On Windows the interpreter is often `py`
rather than `python`; both forms are equivalent.

## From the command line

```powershell
cd etap-speech-assistant-main
python -m evaluation --run-dir ..\output-template\20260809_164356
```

That command finishes with no model and no network. The older
`python evaluate_responses.py ...` form still works: the launcher left in the
package root calls the same entry point.

## As a function

```python
from evaluation import EvaluationConfig, run_evaluation

outcome = run_evaluation(EvaluationConfig(
    run_dir="output-template/20260809_164356",
    judge_models=["qwen2.5:7b-instruct"],      # optional
    progress=print))                            # optional status callback

print(outcome.summary.acceptance_rate)
print(outcome.results[0].relevance["request_coverage"])
outcome.write("output", emit_bibtex=True)       # this is the only disk write
```

`run_evaluation` **neither writes files nor prints**; it returns objects, so a
single number can be read without parsing a report. Field names on
`EvaluationConfig` match the flags, with hyphens turned into underscores
(`--judge-model` → `judge_models` as a list). The command line calls the same
function, so the two paths cannot drift apart.

## Comparing two runs (for parameter tuning)

When a model is run on a large set of input–output pairs, item-by-item human
labelling is not realistic. That is what **paired comparison** is for: it
compares two configurations that answered the same inputs, item by item, with
neither human labels nor a judge model, and reports whether the change improved
the replies.

```powershell
python -m evaluation.comparison --baseline ..\runs\temp08 --contrast ..\runs\temp02
```

```python
from evaluation import ComparisonConfig, compare_runs

outcome = compare_runs(ComparisonConfig(baseline="runs/temp08",
                                        contrast=["runs/temp02"]))
pair = outcome.pairs[0]
print([m.metric.key for m in pair.improvements])
print([m.metric.key for m in pair.regressions])
outcome.write("output")
```

Several variants can share one baseline (`--contrast` is repeatable), so every
point of a parameter sweep is measured against the same reference.

### Why paired

Both runs receive the same prompts, so the comparison can be paired item by
item. Prompt-to-prompt scatter — often larger than the parameter effect —
cancels, and a difference is detectable on a much smaller sample. Pairing is by
user text, not by order, so a reordered or partly failed run does not silently
misalign; the *k*-th occurrence of a repeated prompt is matched with the *k*-th
occurrence on the other side.

### What each report row states

| Column | Meaning |
| --- | --- |
| `delta` | mean paired difference (variant minus baseline) in the metric's own units |
| `95% CI` | percentile bootstrap interval for that mean paired difference |
| `better/worse` | how many items moved for the better, and how many for the worse |
| `effect` | Cliff's δ with the usual band label; empty on a binary metric, where `delta` itself is the effect size |
| `p(adj)` | Holm-adjusted *p*-value, within the metric's own family (`adherence`, `response`, `runtime`) |
| `verdict` | `improved`, `degraded`, `equivalent`, `no detected change`, or `descriptive` |

Test: **McNemar** on a binary (pass/fail) metric, because only items whose
verdict changed carry information; **Wilcoxon signed-rank** on a continuous
metric, because bounded scales are not normal.

### Three properties that make the verdict trustworthy

**Direction is declared, not inferred.** Improvement or degradation is reported
only where it is known which way is better. Word count and sentence count stay
`descriptive`: neither longer nor shorter is inherently better, and the table
does not invent a verdict for them.

**Significance is corrected within each metric family.** Twenty metrics at the
five-percent level have about a two-thirds chance of at least one false hit if
left uncorrected, so Holm's step-down procedure runs on the scored rows — but
separately inside each of the three predeclared families (`adherence`,
`response`, `runtime`), so the number of logged timings cannot dilute the
evidence for a quality claim. The report states, per family, how many tests
were corrected.

**Magnitude is separated from detectability.** On a large sample a difference of
no consequence still reaches significance, so every row carries an effect size,
and where a negligible-change margin is declared an equivalence test can
conclude that *the change does not matter* — a stronger claim than failure to
detect one.

### Flags and output

| Flag | Default | Role |
| --- | --- | --- |
| `--baseline` | – | run directory used as the reference |
| `--contrast` | – | run produced with the changed setting; repeatable |
| `--spec`, `--constraints` | – | applied **identically** to both runs |
| `--judge-model` | – | costly: grades every run |
| `--judge-url` | `localhost:11434` | Ollama endpoint, used only with `--judge-model` |
| `--selfcheck-samples` | 0 | self-consistency resampling on both runs |
| `--alpha` | 0.05 | significance level |
| `--n-boot` | 2000 | bootstrap resamples per metric |
| `--seed` | 0 | seed for bootstrap resampling |
| `--no-check-metrics` | off | do not compare the individual constraint checks |
| `--out-dir` | `./comparison_output` | output directory |
| `--quiet` | off | do not print the report to stdout |

Output: `comparison_report.txt`, `comparison_metrics.csv` (one row per metric,
for machine processing) and `comparison_results.json`. The report header lists
which settings differ between the two `config_used.yaml` files — and warns if
none do, because then the difference is sampling noise.

### What is worth tuning

The comparison runs without a model, so it is cheap on any prompt set: tier-0
checks are decidable, coverage and readability are deterministic, and a rerun
reproduces them. Turning on `--judge-model` grades both runs: that is slow, and
the judge rows inherit the judge's limits, so a close call should not be
decided from those rows alone.

## Evaluating a run grid in one pass

A parameter sweep is not two runs but a grid: one run directory per cell, and
the question is never “how did this run score” but “which setting was better”.
Batch mode reads the whole tree, scores every run identically, builds the
contrasts the grid was made for, and writes every table into one results
directory.

```powershell
python -m evaluation.batch --root ..\results\text-only
```

```python
from evaluation import BatchConfig, run_batch

outcome = run_batch(BatchConfig(root="results/text-only"))
print(outcome.leaderboard()[0]["adherence_item_strict"])
outcome.write("results/evaluation_result")
```

Every run is scored **once**, however many contrasts it appears in. That saves
work; it also guarantees that a run's figures are the same in every table,
which re-evaluating per contrast would not.

### The five contrasts

Grouping is built from the recorded configuration (`config_used.yaml`), not from
the directory name, so a mistyped folder cannot silently produce a wrong
comparison. A contrast is formed only where **exactly one** property differs
between the runs — that is what makes the difference attributable to that
property.

| Contrast | Held fixed | What varies | Baseline |
| --- | --- | --- | --- |
| `parameters` | model, quantization, recognizer | decoding setting (`t`, `seed`) | the greedy (`t=0`) run |
| `quantization` | model, size, setting, recognizer | weight precision | the highest precision (`fp16`) |
| `model_size` | lineage, precision, setting, recognizer | parameter count | the largest model |
| `cross_model` | decoding setting, recognizer | the model | the strongest variant |
| `recognizer` | model and decoding setting | the recognizer (engine, acoustic model, device) | the recognizer used by most of the grid |

The baseline is not a claim that it is best: it is the **reference the change is
measured against**. A negative delta therefore reads as *this is what the
cheaper configuration costs*. The number of groups is a property of the
recorded configurations, not of a fixed grid size. Two decoding settings of one
model, at one precision and one recognizer, form a `parameters` group whose
baseline is the greedy run; a size ladder at one precision and one setting forms
a `model_size` group whose baseline is the largest model of that lineage.

Model contrasts **hold the recognizer fixed**: what the recognizer wrote is the
model's input, so mixing two recognizers into a model contrast would report an
input change as a model effect. The recognizer identity includes the engine,
the acoustic model, the device and the compute precision
(`vosk-small-en-us-0.15-cpu`, `whisper-small-cuda-int8`), because the same
engine with another model recognizes differently. Where the grid spans more
than one recognizer, the group identifier states which one it holds fixed; with
a single recognizer the names are unchanged. The `recognizer` contrast is the
only one in which the two sides' transcripts differ: pairing is therefore by
recording name, not by recognized text, otherwise the items on which the two
recognizers disagree would be the ones dropped.

The size ladder groups by **lineage**, not by exact family name: the sizes on
offer are split across releases (llama3.2 is 1B and 3B, llama3.1 is 8B), so a
ladder that stopped at the family boundary would omit the largest model. Where
a ladder crosses a release, the group's `varying` field says so — parameter
count and release both change there, and the two cannot be separated.

### What it checks before it compares

A paired comparison is about the configuration only if everything else was
held fixed. Batch mode does not assume that: it checks, and it writes every
breach next to the results it affects: the same item set, the same input text,
the same system prompt, constraint definition, recognizer, compute mode,
context window and token cap. The alternative is a difference table that
silently includes a prompt change.

The check runs **per contrast**, and a breach is attributed to the contrast it
affects: a grid that spans two recognizers is not a breach if every model
contrast holds one recognizer fixed. In the `recognizer` contrast the
recognizer and the transcript are the property under study, so the check
requires identity of the **intended question** instead.

### Output

Without `--out-dir`, results are written **beside** `--root`, as
`evaluation_result`: a re-evaluation does not write into the directory it
reads, and the whole comparison is one archivable unit.

| File | Contents |
| --- | --- |
| `summary_report.txt` | the grid's one readable report: inputs, two leaderboards, answer agreement, strata, contrasts, source list |
| `leaderboard.csv` | one row per run: the configuration and every headline metric side by side |
| `comparison_index.csv` | every metric of every contrast, with *p*-value and verdict, for machine processing |
| `input_reference.csv` | the shared input per recognizer: intended text, recognized text, per-item error rates |
| `input_quality_strata.csv` | breakdown by run × input quality |
| `input_quality_impact.csv` | rank correlation of recognition error with every response metric |
| `all_items.csv` | every item of every run in one table (number of runs × *n*), for a pivot or a figure |
| `batch_manifest.json` | which runs were read, with which settings, and where their results were written |
| `runs/<cell>/` | per-run report, per-item CSV, JSON, and a copy of `config_used.yaml` |
| `comparisons/<kind>/<group>/` | the full paired-comparison table of one contrast |

### Flags

| Flag | Default | Role |
| --- | --- | --- |
| `--root` | – | directory holding the runs; searched recursively for a transcript |
| `--out-dir` | `<root>/../evaluation_result` | output directory |
| `--group` | all five | build only the named contrast; repeatable |
| `--spec`, `--constraints` | – | applied **identically** to every run |
| `--answer-key` | – | corpus metadata table (CSV) with reference answers, applied **identically** to every run |
| `--embedding-model` | – | sentence-transformers model for embedding similarity, in every run |
| `--judge-model`, `--selfcheck-samples` | – | costly: runs on every cell |
| `--no-latency` | off | do not read the latency logs |
| `--latency-warmup` | the value recorded in the run's `log_averages.json` | leading items to drop from the timing aggregates; quality scores are unaffected |
| `--no-check-metrics` | off | do not compare the individual constraint checks |
| `--alpha`, `--n-boot`, `--seed` | 0.05, 2000, 0 | as in paired comparison |
| `--quiet` | off | do not print the report to stdout |

## Dependencies

`requests`, `PyYAML`, `numpy` — install with
`pip install -r evaluation/requirements.txt`. `scipy` and
`sentence-transformers` are optional and detected at runtime. Model-backed
tiers need a locally running Ollama server; if it is absent the run does not
stop, it only records the skipped tier.

## Package layout

| File | Role |
| --- | --- |
| `pipeline.py` | `EvaluationConfig`, `EvaluationOutcome`, `run_evaluation()` — orchestration |
| `cli.py`, `__main__.py` | command-line layer over `run_evaluation` |
| `comparison.py` | paired comparison of two runs, with its own command line |
| `batch.py` | scoring a run grid and building the contrasts, with its own command line |
| `constraints.py` | tier 0: decidable checks |
| `asr.py` | recognizer fidelity against the intended question: WER, CER, content recall, strata |
| `latency.py` | per-item stage times and resource cost from the run's latency log |
| `relevance.py` | tier 1: coverage and overlap |
| `factuality.py` | tier 2: audit atoms, self-consistency, atomic claims |
| `judge.py` | tier 3: rubric, panel, pairwise contrast |
| `agreement.py`, `stats.py` | tier 4 and the statistical tools |
| `aggregation.py` | per-item scores, run-level summary, acceptance policy |
| `reporting.py` | report, CSV, JSON |
| `references.py` | method → literature → validation-status registry |
| `loaders.py` | transcripts, scenario spec, corpus answer key, and run artefacts |
| `textutils.py` | sentence splitting, normalization, syllable counts — shared by the metrics |
| `ollama_client.py` | HTTP layer to the local model: retries, JSON recovery |
| `selftest.py` | offline checks against worked or published values |
| `fake_ollama.py` | protocol-faithful stub server for the model-backed tiers |
| `requirements.txt` | the three required dependencies, plus the two optional ones in comments |

Two data directories ship with the package. `rubrics/` **is not optional**: the
code's defaults refer to it.

| File | Role |
| --- | --- |
| `rubrics/constraints_short_opener.yaml` | definition of the 15 tier-0 checks (default of `--constraints`) |
| `rubrics/quest_therapy_v1.yaml` | the 15-dimension judge rubric (default of `--rubric`) |

`examples/` is a template: copy and edit it; the package does not need it to
run.

| File | Role |
| --- | --- |
| `examples/scenarios_open_domain_smoke.yaml` | full scenario spec with reference answers and forbidden patterns |
| `examples/scenarios_minimal_categories.yaml` | variant without references: category and expected behaviour only |
| `examples/scenarios_therapy_template.yaml` | therapy template; this file lists **every** field that is read and the 11 category names |
| `examples/human_annotations_example.csv` | tier-4 input format: `item_id,rater_id,dimension,score` |

## Tiers

| Tier | What it measures | When it runs |
| --- | --- | --- |
| 0 | prompt adherence, readability | always, no model |
| – | recognizer fidelity (WER, CER, content recall, strata) | when the transcript has `ori_text` |
| – | stage times and resource cost | when a `latency_log_*.csv` sits beside the run |
| 1 | relevance, reference overlap, answer agreement | always; overlap and agreement need `--answer-key` or `--spec` |
| 2 | factual commitments, hallucination | audit atoms always; the rest optional |
| 3 | rubric grading | with `--judge-model` |
| 4 | rater agreement, calibration | with `--human-annotations` |

### Input quality and runtime

The two unnumbered rows are not about the reply. They describe **what the model
was given** and **what the reply cost**. Neither needs a model or a reference
answer.

Recognizer fidelity is measured against the transcript's `ori_text`, i.e. the
question that was intended to be spoken. It is not a quality score: it bounds
what the model *could* answer. On a numeric reference, several spoken
realizations (cardinal, year, digit-by-digit) are tried and the best match
counts, so the recognizer is not charged for hearing “1990” spoken aloud. Items
are placed in three WER strata — `clean` (exact), `mild` (at most one third of
the reference words damaged), `severe` (more than that) — and the report also
gives prompt adherence by stratum. Coverage should not be compared across
strata: a correct closed answer does not repeat the question, so its coverage
is low for a reason that does not involve the model.

Runtime comes from the run's own `latency_log_*.csv`, joined per item by
recording name — not by order, so a broken run cannot shift the rest. The
stages overlap and do not start from the same instant, so they cannot be
summed: `stt` is the recording arriving at speaking pace (on file input, the
length of the audio, not a compute cost), `ttfa` runs from the end of speech to
the first emitted audio — that is the wait the user perceives — and
`e2e_response_ready` is the whole item including the recording, so it mainly
reflects speech length. When models are compared, `ttfa` and token throughput
are the informative columns.

Resource columns (device memory, GPU utilisation) are near-constant within a
run, so they are summarized at run level rather than compared item by item.

## Flags (single run)

| Flag | Default | Role |
| --- | --- | --- |
| `--run-dir` | – | run directory; transcripts, config and system prompt are read from it |
| `--transcripts` | – | standalone transcript file when the rest of the run artefacts are missing |
| `--spec` | – | scenario spec: category, reference answer, required/forbidden content |
| `--answer-key` | – | corpus metadata table (CSV): `answer` / `plausible_answers` by `is_impossible` |
| `--system-prompt` | from the run | overrides the system prompt used for judging and resampling |
| `--label` | run directory name | label written into the report |
| `--constraints` | `constraints_short_opener.yaml` | definition of the tier-0 checks |
| `--max-tokens` | from config | `num_predict` cap used for truncation detection |
| `--embedding-model` | – | e.g. `all-MiniLM-L6-v2`; downloads on first use |
| `--selfcheck-samples` | 0 | self-consistency resamples; 0 = off |
| `--selfcheck-model` | from config | must be **the model that produced the replies** |
| `--selfcheck-temperature` | 0.8 | resampling temperature |
| `--fact-precision` | off | atomic-claim decomposition and labelling |
| `--max-claims` | 10 | claim cap per reply |
| `--judge-model` | – | judge Ollama tag; repeatable, several models form a panel |
| `--judge-url` | `localhost:11434` | Ollama endpoint |
| `--judge-samples` | 1 | samples per dimension; above 1 approximates a G-Eval expected score |
| `--judge-timeout` | 300 | per-request timeout in seconds |
| `--rubric` | `quest_therapy_v1.yaml` | rubric definition |
| `--compare-run-dir` | – | second run for A/B judging, in both presentation orders |
| `--pairwise-dimension` | relevance, spoken_comprehensibility, factual_accuracy | rubric dimension for that A/B comparison; repeatable |
| `--human-annotations` | – | long-format CSV: `item_id,rater_id,dimension,score` |
| `--min-quality` | 3.5 | acceptance threshold on the quality composite |
| `--min-safety` | 4.0 | threshold on the worst safety dimension |
| `--loose-constraints` | off | gate on the loose verdict, ignoring formatting-only failures |
| `--no-constraint-gate` | off | do not require constraint adherence for acceptance |
| `--out-dir` | `<run-dir>/evaluation` | output directory |
| `--emit-bibtex` | off | BibTeX for the methods actually used |
| `--seed` | 0 | seed for bootstrap resampling |
| `--quiet` | off | do not print the report to stdout |

> Acceptance thresholds must be **fixed before measurement**: they are written
> into the report header, and tuning them afterwards makes the acceptance rate
> meaningless.

## Output (single run)

| File | Contents |
| --- | --- |
| `evaluation_report.txt` | readable report: sections, source list, interpretation limits |
| `evaluation_items.csv` | one row per item, with every metric |
| `evaluation_results.json` | full structured output |
| `evaluation_references.bib` | BibTeX, with `--emit-bibtex` |

## Computed values and how to read them

The tables below follow the columns of `evaluation_items.csv`, in file order.
A column is present only if the tier that produces it ran — each section states
when — so a model-free run's CSV is shorter, not incomplete.

**Status:** `verifiable` = decided by construction, `validated` = a validated
instrument or published statistical theory, `established` = a widely used
method with published human correlation, `surrogate` = a deliberate
simplification of a published method. Where the source is `–`, there is no
paper behind it: a local but deterministic computation.

### Identification (always)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `item_id` | stable identifier; from the recording name, the scenario spec, or a serial | – | verifiable |
| `filename` | recording name; the join key to the latency log and across runs | – | verifiable |
| `category` | scenario category; this decides which checks and rubric dimensions apply | – | verifiable |
| `safety_critical_item` | whether the item is safety-critical | – | verifiable |
| `stt_text`, `llm_text` | the scored input and reply, verbatim | – | verifiable |
| `ori_text` | the intended spoken question, when the transcript has it | – | verifiable |
| `reference_answer` | the answer the item is scored against, when an answer key or spec supplied one | – | verifiable |
| `answer_unsupported` | whether the corpus marks the item as unanswerable | – | verifiable |

### Tier 0 — decidable prompt adherence (always)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `constraint_item_pass_strict` | whether **every** hard constraint held; the headline adherence indicator | Zhou et al., 2023 (IFEval) | verifiable |
| `constraint_item_pass_loose` | the same after formatting transforms; if it differs from strict, the failure is formatting-only | Zhou et al., 2023 (IFEval) | verifiable |
| `constraint_check_rate_strict` | fraction of checks passed; inapplicable checks leave the denominator | Jiang et al., 2024 (FollowBench) | verifiable |
| `constraint_check_rate_loose` | the same with the loose verdict | Jiang et al., 2024 (FollowBench) | verifiable |
| `constraint_failures_strict` | names of the failed checks, semicolon-separated; an audit trail | – | verifiable |
| `constraint_failures_loose` | the same with the loose verdict | – | verifiable |

The strict/loose pair follows IFEval's dual evaluation. The 15 checks: non-empty
reply, opening sentence ≤ 5 words, opening sentence ends with a full stop, 0–2
detail sentences, 1–3 sentences in total, English, no simulated dialogue, no
follow-up question (advisory), completeness, token-cap truncation, word count
(advisory), spoken duration (advisory), no markup, and required and forbidden
content from the scenario spec.

### Descriptive and readability metrics (always)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `word_count`, `sentence_count`, `char_count` | length | – | verifiable |
| `mean_words_per_sentence` | mean sentence length | – | verifiable |
| `opening_words` | word count of the first sentence; the measured value of the opener constraint | – | verifiable |
| `estimated_tokens` | an **estimate**, not a measured token count; used as a truncation suspicion | – (heuristic in place of a tokenizer) | surrogate |
| `estimated_spoken_seconds` | estimated spoken duration at 2.5 words/s | – (heuristic) | surrogate |
| `flesch_reading_ease` | higher = easier | Flesch, 1948 | validated |
| `flesch_kincaid_grade` | difficulty in US school-grade years | Kincaid et al., 1975 | validated |

### Recognizer fidelity (when `ori_text` is present)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `stt_wer` | word error rate against the intended question; among spoken variants of numbers, the best match counts | Levenshtein, 1966 | verifiable |
| `stt_cer` | character error rate; a misspelled word costs less here than in WER | Levenshtein, 1966 | verifiable |
| `stt_substitutions`, `stt_deletions`, `stt_insertions` | errors by type | Levenshtein, 1966 | verifiable |
| `stt_content_recall` | how many of the content (non-function) words survived recognition | – | verifiable |
| `stt_exact_match` | whether recognition matches the reference verbatim | – | verifiable |
| `stt_stratum` | `clean`, `mild` or `severe` from WER | Wang et al., 2003 | verifiable |

WER is not a quality score but a bound: it says how *different* a question the
model received (Wang et al., 2003).

### Runtime and resources (when a latency log is present)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `lat_stt`, `lat_stt_endpoint_delay` | arrival of the recording, and finalization after the end of speech; controls, not about the model | – | verifiable |
| `lat_llm_prompt_eval`, `lat_llm_ttft`, `lat_llm_first_chunk_fill`, `lat_llm_ttfc` | prompt evaluation, first token, fill of the first speakable chunk, first speakable unit | – | verifiable |
| `lat_tts_first_chunk` | blocking synthesis of the first audio chunk | – | verifiable |
| `lat_ttfa` | from the end of speech to the first emitted audio: the wait the user perceives | Walker et al., 1997 (PARADISE) | verifiable |
| `lat_llm_eval`, `lat_tts_total` | full generation and synthesis time; both grow with reply length | – | verifiable |
| `lat_e2e_response_ready` | wall-clock span of the item, including the recording | – | verifiable |
| `llm_prompt_tokens`, `llm_eval_tokens`, `llm_tokens_per_sec` | the engine's own token counts and throughput | – | verifiable |
| `llm_chunk_count` | number of speakable chunks released | – | verifiable |
| `tts_audio_ms` | duration of the synthesized audio | – | verifiable |
| `stt_rtf`, `input_duration_ms`, `trailing_silence_ms` | recognizer real-time factor, input length, trailing silence; from the log extras | – | verifiable |

The report also gives the median and the 95th percentile, because a spoken
assistant is judged by the late replies (Dean and Barroso, 2013). Device memory
and GPU utilisation are summarized at run level (`llm_vram_mb`,
`llm_model_vram_mb`, `gpu_util_percent`, …); they do not appear as item-level
CSV columns.

### Tier 1 — relevance (`reference_*` columns need `--answer-key` or `--spec`)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `request_coverage` | IDF-weighted coverage of the **recognized** question's content words; no reference answer needed | – (IDF-weighted overlap) | surrogate |
| `intent_coverage` | the same against the **intended** question, when `ori_text` is present | – (IDF-weighted overlap) | surrogate |
| `coverage_intent_gap` | the difference: how much of the interaction the recognizer removed | – | verifiable |
| `echo_ratio` | how much of the question is repeated back; a high value suggests an empty echo | – | verifiable |
| `request_response_cosine` | embedding similarity of question and reply; only with `--embedding-model` | Zhang et al., 2020 (related method) | surrogate |
| `intent_response_cosine` | the same against the intended question; needs `ori_text` and `--embedding-model` | Zhang et al., 2020 (related method) | surrogate |
| `answer_presence` | 0/1: whether the reference-answer span occurs in the reply (after SQuAD normalization, contiguously) | Chen et al., 2017 | established |
| `reference_exact_match` | 0/1: whether the normalized reply *is* the reference answer | Rajpurkar et al., 2016 (SQuAD) | established |
| `reference_answer_words` | length of the reference answer in words: this is what separates a short span from a paragraph extract | – | verifiable |
| `reference_token_f1` | token-level overlap with the reference answer | Rajpurkar et al., 2016 (SQuAD) | established |
| `reference_rouge_1` | unigram overlap | Lin, 2004 | established |
| `reference_rouge_l` | longest-common-subsequence overlap | Lin, 2004 | established |
| `reference_cosine` | embedding similarity to the reference; only with `--embedding-model` | Zhang et al., 2020 (related method) | surrogate |

With several reference answers, the best match is kept. Overlap metrics
correlate poorly with human judgement in dialogue (Liu et al., 2016), so they
are for **screening**, not as a quality score.

For an assistant that answers in a sentence, `answer_presence` is the
informative one of the three agreement metrics: `reference_exact_match` is
structurally near zero because the model does not return the bare span, and
`reference_token_f1` is driven down by every extra word in the sentence.
`answer_presence` is an **upper bound** on correctness: a reply that quotes the
span while asserting something else still scores, a correct paraphrase does
not — graded correctness needs a judge model (`--judge-model`) or a human
round.

#### The corpus's own answer key (`--answer-key`)

Reference answers come from the corpus metadata table (CSV), not from a
hand-written spec, so the `reference_*` columns trace back to the published
dataset. The columns follow the SQuAD 2.0 convention (Rajpurkar et al., 2018):

| Item state | Which column is the reference | What a match supports |
| --- | --- | --- |
| `is_impossible=FALSE` | `answer`: the annotated correct span | on a match, the reply was correct |
| `is_impossible=TRUE` | `plausible_answers`: what the annotator found acceptable | on a match, the reply *resembles* what an annotator would have said — not correctness |

The key is joined by **recording name** (item identifier, filename, then
question text, in that order), because a misheard transcript must still be
scored against the answer of the item the system was given. If both `--spec`
and `--answer-key` are present, the spec wins: it was written for one
evaluation, the key describes the whole corpus.

The loader repairs two export artefacts, both because the silent alternative
would bias the measurement: a leftover quote pair around the answer is stripped,
and if a paragraph has leaked into the answer column the span after the closing
quote is taken as the answer. A row with no answer in either column is left
without a reference and is not entered into the key.

The run summary (`answer_accuracy`, and the `leaderboard.csv` `*_short_span` /
`*_plausible` columns) reports three subsets separately, because they do not
support the same claim: (1) an answerable item with a short span of at most 8
words — here presence is evidence of correctness; (2) an unanswerable item,
with only a plausible answer; (3) an answerable item whose span is a paragraph
extract, which a few-sentence reply cannot reproduce. A pooled rate would
represent none of them correctly.

### Tier 2 — factual commitments and hallucination

`audit_atoms` is always produced. The `selfcheck_*` columns appear when
`--selfcheck-samples` > 0, the `factprecision_*` columns when `--fact-precision`
is set.

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `audit_atoms` | extracted years, quantities, names and their counts: a checklist for human audit | Ji et al., 2023 (taxonomy) | verifiable |
| `selfcheck_kernel` | `token_f1_surrogate` or `embedding_cosine`: which support measure ran | Manakul et al., 2023 | surrogate |
| `selfcheck_samples` | how many resamples were compared | Manakul et al., 2023 | established |
| `selfcheck_mean_inconsistency` | 0–1; how poorly the resamples support the reply's sentences | Manakul et al., 2023 | established |
| `selfcheck_max_inconsistency` | the worst sentence | Manakul et al., 2023 | established |
| `selfcheck_flagged_sentences` | sentences above the threshold, verbatim | Manakul et al., 2023 | established |
| `selfcheck_error` | why it was skipped, when it was | – | verifiable |
| `factprecision_n_claims` | number of atomic claims judged | Min et al., 2023 (FActScore) | established |
| `factprecision_supported`, `factprecision_unsupported`, `factprecision_unverifiable` | counts of the three labels | Min et al., 2023 (FActScore) | established |
| `factprecision_precision` | fraction of supported claims | Min et al., 2023 (FActScore) | established |
| `factprecision_knowledge_source` | whether a reference answer was the knowledge source, or the model's own knowledge | – | verifiable |
| `factprecision_unsupported_claims` | claims labelled unsupported | Min et al., 2023 (FActScore) | established |
| `factprecision_error` | why it was skipped, when it was | – | verifiable |

Two limits: self-consistency catches hallucination that comes from
**uncertainty**; a confident, consistently repeated falsehood stays invisible.
The default support measure is token overlap, not the published BERTScore/NLI
kernel, so absolute values are not comparable with published SelfCheckGPT
figures — they rank items within a run. On `factprecision_precision` without a
reference, the judge's own knowledge is the yardstick, so
`factprecision_knowledge_source` must be read before the number is reported.

### Tier 3 — rubric (only with `--judge-model`)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `judge_<dimension>` | 15 dimensions on a 1–5 scale, with anchors | Liu et al., 2023 (G-Eval); Kim et al., 2024 (Prometheus 2) | established |
| `judge_<dimension>_panel_spread` | on a panel, the largest disagreement; a large value means an unreliable dimension | Verga et al., 2024 | established |
| `quality_composite` | weighted mean of the **non**-safety dimensions | Tam et al., 2024 (QUEST) | validated |
| `safety_minimum` | the worst safety dimension; it gates, it is not averaged | Singhal et al., 2023 | validated |

The rubric is built on the QUEST frame, but the weights are a local decision,
not part of the published instrument. Pairwise comparison with
`--compare-run-dir` runs every pair in both orders; a verdict that flips with
order is a tie, because that is position bias, not a preference (Zheng et al.,
2023).

### Acceptance (always)

| Column | Meaning | Source | Status |
| --- | --- | --- | --- |
| `accepted` | verdict of the predeclared policy | Gallifant et al., 2025 (TRIPOD-LLM) | verifiable |
| `acceptance_reasons` | why it failed; empty on a pass | – | verifiable |

If an input the decision needs is missing — for example the quality composite
on a run without a judge — `accepted` stays empty; it is not defaulted to fail.

### Tier 4 — agreement and calibration (only with `--human-annotations`)

These are run-level values. They do not appear in the per-item CSV; they are in
the report and the JSON.

| Metric | Reading | Source | Status |
| --- | --- | --- | --- |
| `percent_agreement` | raw agreement among rater pairs; a baseline | – | verifiable |
| `krippendorff_alpha_ordinal` / `_nominal` | handles missing cells and more than two raters | Krippendorff, 2018 | validated |
| `cohen_kappa`, `fleiss_kappa` | chance-corrected agreement; band labels follow Landis and Koch | Cohen, 1960; Fleiss, 1971; Landis and Koch, 1977 | validated |
| `gwet_ac1` | read this instead of κ on unbalanced categories | Gwet, 2008; Feinstein and Cicchetti, 1990 | validated |
| `icc_2_1`, `icc_2_k` | two-way random model; label follows Koo and Li | Shrout and Fleiss, 1979; Koo and Li, 2016 | validated |
| `judge_bias_vs_human`, `judge_mae_vs_human` | the judge's systematic offset and mean absolute error | – | verifiable |
| `judge_human_spearman`, `judge_human_kendall_tau` | rank correlation of judge and human | – (rank correlation) | validated |
| PPI estimate | a valid interval even when the judge is systematically wrong | Angelopoulos et al., 2023; Boyeau et al., 2024 | validated |

### Aggregation

Every run-level mean has a percentile bootstrap interval (Efron and Tibshirani,
1993), which does not assume normality. Also available: Cliff's δ (1993), TOST
equivalence (Lakens, 2017) and Holm correction (1979); two-run comparison adds
the Wilcoxon signed-rank test (1945) and McNemar's test (1947).

Holm correction runs **per metric family** — `adherence`, `response`,
`runtime` — not over the whole table. The families are declared in the code in
advance, so they are not chosen after seeing the results. The reason: most of
the roughly forty metrics are milliseconds, and a single pooled correction
would let the number of logged timings dilute the evidence a quality claim
needs — and that number is a property of how finely the pipeline is
instrumented, not of the question.

## Literature used

Status: `verifiable` = decided by construction, `validated` = a validated
instrument or published statistical theory, `established` = a widely used
method with published human correlation, `surrogate` = a deliberate
simplification of a published method.

| Method | Source | Status |
| --- | --- | --- |
| verifiable instruction following | Zhou et al., 2023 (IFEval) | verifiable |
| token-level F1 | Rajpurkar et al., 2016 (SQuAD) | established |
| ROUGE-1, ROUGE-L | Lin, 2004 | established |
| limits of overlap metrics | Liu et al., 2016 | validated |
| readability | Flesch, 1948; Kincaid et al., 1975 | validated |
| self-consistency | Manakul et al., 2023 (SelfCheckGPT) | established |
| token-overlap support kernel | – (in place of the published BERTScore/NLI kernel) | surrogate |
| atomic factual precision | Min et al., 2023 (FActScore) | established |
| rubric grading | Liu et al., 2023 (G-Eval); Kim et al., 2024 (Prometheus 2) | established |
| judge panel | Verga et al., 2024 | established |
| position-bias control | Zheng et al., 2023 | established |
| health-evaluation frame | Tam et al., 2024 (QUEST) | validated |
| empathy | Sharma et al., 2020 (EPITOME) | validated |
| over-refusal | Röttger et al., 2024 (XSTest) | established |
| clinical overreach | Singhal et al., 2023 | validated |
| Krippendorff's α | Krippendorff, 2018 | validated |
| Cohen's κ, Fleiss's κ | Cohen, 1960; Fleiss, 1971 | validated |
| Gwet's AC1, prevalence paradox | Gwet, 2008; Feinstein and Cicchetti, 1990 | validated |
| ICC | Shrout and Fleiss, 1979; Koo and Li, 2016 | validated |
| prediction-powered inference | Angelopoulos et al., 2023 | validated |
| bootstrap interval | Efron and Tibshirani, 1993 | validated |
| Cliff's δ, TOST, Holm | Cliff, 1993; Lakens, 2017; Holm, 1979 | validated |
| paired rank test, paired binary test | Wilcoxon, 1945; McNemar, 1947 | validated |
| edit distance, word and character error rate | Levenshtein, 1966 | verifiable |
| limits of WER as a proxy for understanding | Wang et al., 2003 | established |
| the 95th percentile as the user-perceived delay | Dean and Barroso, 2013 | established |
| cost measurement for spoken dialogue systems | Walker et al., 1997 (PARADISE) | validated |

`--emit-bibtex` writes BibTeX entries for the methods actually used, so the
bibliography does not swell with unused items.

## Limits that a paper using these numbers should also state

A judge score is a **screening tool, not a validated instrument**; without
human calibration it should not be published as a quality estimate.
Self-consistency values rank items within the run; they are not comparable with
published SelfCheckGPT figures. The readability formulae were validated on
written text; intelligibility of synthesized speech needs a listening test
(ITU-T P.85 or P.808). Finally, coverage bounds every claim: a figure computed
on open-domain items does not support a statement about therapy scenarios.

## Self-test

```powershell
python -m evaluation.selftest
```

The suite runs without a network. It checks the statistics against worked or
published values, because a wrong coefficient still returns a plausible number.
The printed line `OK: N checks passed` is the current count (224 as of this
writing).
