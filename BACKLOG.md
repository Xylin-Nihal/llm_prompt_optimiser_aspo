# ASPO Backlog: what is left in each module

This file lists the work still open in each module, and the known limitations. The plan is to build an MVP first and then come back here.

The **MVP** is the roadmap's "usable v1" (weeks 8–9 in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §9). Every module M0–M8 exists in a basic working form, and an OpenAI SDK pointed at the proxy works end to end.

How to read this file:

- **[MVP]** — needed for the MVP to work correctly. Do it while building, not afterwards.
- **[post-MVP]** — finish once the MVP runs.
- **[paper]** — needed only for the evaluation or the paper text.
- **Limitation** — a known weakness. Fix it, or state it in the paper.

Tick the boxes as items are done, and add new items as modules are built.

*Last updated: 2026-11-27 · built: M0, M1 (incl. tiers + intake), M2 v1, M6 · next: M3 pilot scorer + M5 loop*

---

## M0 — Tokenizer layer `aspo/tokenizers.py` ✅

**Open work**

- [ ] **[MVP]** Get exact Llama-3.1 counts. Accept the licence for `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face, then set `HF_TOKEN`. Until then, Llama counts are approximated, flagged `exact=False`. Llama is one of the three target models, so this has to be fixed before any results are produced.
- [ ] **[post-MVP]** Calibrate the Claude and Gemini approximations with `calibrate()`. Steps:
  1. Collect about 100 prompts that cover English, code, JSON and non-English text.
  2. Get exact counts from Anthropic `messages.count_tokens` and Gemini `countTokens`.
  3. Fit the factor and record it, with its error, in the results.
- [ ] **[post-MVP]** Optional: a `register_tokenizer` backend that calls a provider's counting endpoint directly, for exact Claude and Gemini counts. Use it for evaluation only, never inside the loop, because it costs a network call per count.
- [ ] **[post-MVP]** `count_messages(messages, model)`: count with chat-template overhead, for absolute per-request cost. Savings do not need it, because the overhead cancels out.
- [ ] **[post-MVP]** Wire the cache-read price (`ModelPrice.cache_read_per_token`) into the `prefix_freeze` cost estimate (RQ5).
- [ ] **[paper]** Pin the LiteLLM version used for the reported prices, and state the price date in the paper.

**Limitations**

- Approximated counts (Claude, Gemini, unknown models) are o200k_base × a factor, not the real tokenizer. Calibrate the factor, or report the error.
- Counts cover raw text only, with no BOS token and no chat-template tokens.
- A GPT-like model name that tiktoken does not know is assumed to use o200k_base, and is flagged.
- The HF tokenizer is downloaded on first use (network, cached locally afterwards). The repo mapping covers only Llama-3.x and Qwen2/2.5; other open models fall back to the approximation.
- `import litellm` takes about 6 s. It happens lazily, on the first price lookup.

---

## M1 — Intake + Segmenter `aspo/segmenter.py`, `aspo/protected.py` ✅

**Open work**

- [x] **Two-tier protection mapping** (done 2026-11-26). `Tier` on every `ProtectedSpan`, `span_tier(kind, role)`, `Segment.hard_spans / soft_spans / is_hard() / removal_units()`. OTHER segments count as hard. A soft span that overlaps a hard span is promoted to hard.
- [x] **Intake checks** (done 2026-11-26, `aspo/intake.py`). Empty prompts raise `EmptyPromptError`. A prompt longer than the model's context window (or `max_tokens`) returns `optimise=False, reason="too_long"` and is sent unchanged.
- [ ] **[post-MVP]** Decide whether section labels ("Context:", "Question:") should be hard. Today they are ordinary removable words; the validator's Q check only guards that a QUESTION/FORMAT segment is not removed entirely.
- [ ] **[post-MVP]** Simple language check. Non-English prompts get the "safe" preset, which removes filler only.
- [ ] **[post-MVP]** Install spaCy plus `en_core_web_sm` in `.venv` and re-run the tests with NER on. Today, entities come from rules only.
- [ ] **[paper]** Measure segmenter role accuracy on a small hand-labelled set. Optional, but it backs the "role-aware" claim.

**Limitations**

- Role assignment and span detection are rule-based and English-only (the lexicon and regexes).
- Without spaCy, ENTITY detection is weaker: capitalised-phrase rules only.
- Constraint and format detection depends on cue words. An unusual phrasing of a constraint can be missed, and a missed hard span is not protected.
- Negation spans protect the whole clause, so entities and numbers inside a negated sentence in a document become hard. This is safe, but lowers compression on negation-heavy documents.
- Intake counts text only (no chat-template tokens, no images), so a prompt just under the context window can still be rejected by the provider.
- If a model's context window is not in LiteLLM's table, the length is not checked; `notes` says so.

---

## M2 — Intent & Complexity Analyzer `aspo/analyzer.py` ✅ v1

**Open work**

- [x] Length factor in T_M tokens (done 2026-11-25, `AnalyzerConfig.target_model`).
- [ ] **[MVP]** Pass the target model from the pipeline, so that the analyzer always uses token length in practice.
- [ ] **[post-MVP]** Prototype-embedding scorer, formula (2b):
  - 5–10 prototype prompts per task, embedded with `all-MiniLM-L6-v2`;
  - the cosine to each task centroid is added to the rule logits;
  - optional dependency, falling back to rules only.
- [ ] **[paper]** 400-prompt evaluation set from LMSYS-Chat-1M or ShareGPT, labelled independently by two people (`data/analyzer_labels.csv`). Report Cohen's κ and macro-F1, rules-only vs rules + prototypes.
- [ ] **[post-MVP]** Re-tune the complexity weights and `long_prompt_tokens` (5,000) on the validation split. They are hand-set today.

**Limitations**

- v1 uses hand-weighted cues, not learned weights. Accuracy has not been measured yet (it needs the 400-prompt set).
- English-only cue lexicon.
- The complexity score C sets the validator threshold τ(C), so analyzer errors flow into validator strictness.

---

## M3 — Pilot removal scorer `aspo/scorer.py` ⏳

**MVP scope:** an IDF-style word scorer (or leave-one-out drift, formula (1)) that returns one score per removable word, respecting hard spans and atomic soft spans.

**Open work after the MVP version**

- [ ] **[post-MVP]** Implement both variants (IDF and leave-one-out drift) and compare them.
- [ ] **[paper]** Use it as the "pilot loop without the learned policy" baseline.

---

## M4 — Learned removal policy `aspo/policy.py`, `training/` ⏳

**MVP scope:** can be left out of the MVP. The loop runs with the M3 scorer. Plan order: the pilot comes first, then the go / no-go decision, then the policy.

**Open work**

- [ ] **[post-MVP]** Start from LLMLingua-2 mBERT-base. Average subword scores per word, force hard-span words to keep, and score each soft span as one unit.
- [ ] **[post-MVP]** One-shot REINFORCE training, with the reward gate H ∧ Q ∧ cos ≥ τ(C). Develop on the laptop; run full training on Kaggle.
- [ ] **[paper]** Reward-weight grid on the validation split. Comparison with the pilot scorer. Tokenizer transfer test (RQ2).
- [ ] **[post-MVP]** Optional XLM-RoBERTa-large run.

**Limitation to state:** the policy is trained one-shot but run iteratively.

---

## M5 — Normalize and removal loop `aspo/optimizers.py` ⏳

**MVP scope:** normalize (whitespace, filler phrases, JSON minify), plus the loop with the M3 scorer: remove the k lowest-scored words, validate with f = 1, restore and stop on the first failure.

**Open work after the MVP version**

- [ ] **[paper]** Batch-size sweep k ∈ {1, 4, 16, 64} and the 5% default. Validator-frequency sweep f ∈ {1, 2, 4, 8}.
- [ ] **[paper]** Measure how often a later step would have passed after the first failure (the stop-point assumption, formula (11)).
- [ ] **[post-MVP]** Optional lexical rewrite (M5.3), as a separate ablation.

**Limitation to state:** the stop-point assumption is not guaranteed.

---

## M6 — Validator `aspo/validator.py` ✅

**Open work**

- [ ] **[MVP]** **Decide what to do about the NLI check (N).** Measured on 2026-11-27: the backward direction (does the shortened instruction entail the original?) scores imperatives very low. Dropping "following" from "Summarize the following article in 3 bullet points" scored 0.011, while dropping politeness phrases passed (0.96–0.99). At p ≥ 0.8, N will block almost every edit to instructions, so in practice only context is compressed. Options:
  - keep it as is, and report N as conservative;
  - lower the threshold, tuned on the validation split;
  - forward direction plus the similarity check only;
  - rewrite instructions as statements ("The task is to …") before running NLI;
  - try a larger NLI model (`nli-deberta-v3-base`).
- [ ] **[post-MVP]** Speed up S. About 40–50 ms per step comes mostly from one MiniLM call per changed chunk; batch the candidates of several steps, or move the vector maths to numpy or torch.
- [ ] **[paper]** Validator precision and recall against human labels. Validator on vs off. Cosine-only vs cosine + NLI. Frequency sweep f ∈ {1, 2, 4, 8}.
- [ ] **[post-MVP]** Tune τ0, τ1 and the NLI threshold on the validation split.

**Limitations**

- The validator is a proxy for answer quality, not a guarantee.
- DeBERTa reads at most 512 tokens, so a very long instruction segment is only partly checked.
- H checks that each hard span appears verbatim and in order inside its segment. It does not check that the hard span still sits next to the same words.
- Soft-span retention is measured by substring match, so the same text appearing elsewhere in the segment can count as "kept".
- The chunked-mean embedding is a design choice (all-MiniLM-L6-v2 truncates at 256 word pieces). State it in the paper's formula for e(·).
- Cold start: MiniLM takes about 2 s and DeBERTa about 10 s to load on first use. Load them once per process (proxy start-up).

---

## M7 — Model adapter `aspo/adapters.py` ⏳

**MVP scope:** LiteLLM calls to OpenAI and Ollama, streaming passed through, and the target model handed to M0.

**Open work after the MVP version**

- [ ] **[post-MVP]** Test with Anthropic and Gemini.
- [ ] **[paper]** Response caching at temperature 0, for reproducible evaluation runs.

---

## M8 — Delivery surfaces ⏳

**MVP scope:**

- `aspo.optimize()` library call;
- the OpenAI-compatible FastAPI proxy with `X-TokenWise-*` headers;
- the CLI `python -m aspo optimize`.

**Open work after the MVP version**

- [ ] **[post-MVP]** Streamlit dashboard: diff view, roles, per-step validator results, preset slider.
- [ ] **[post-MVP]** Safeguards: overhead gate, opt-in local logging, `prefix_freeze`.
- [ ] **[post-MVP]** Update `aspo/__main__.py` (it runs the segmenter today) into a CLI with subcommands.

---

## Evaluation and paper (after the MVP)

- [ ] `eval/` harness: `datasets.py`, `baselines.py`, `metrics.py`, `run_benchmark.py`, writing `results/*.csv`.
- [ ] Baselines: random, truncation, Selective Context, LLMLingua, LongLLMLingua, LLMLingua-2, LLM-DCP, TACO-RL.
- [ ] Statistics: 3 seeds, bootstrap CIs, paired bootstrap and McNemar tests, Pareto curves.
- [ ] Replace the placeholder numbers in the paper draft with results from `results/` only. Apply the paper fixes (the plan refers to `PAPER_REVISIONS.md`, which is not in the repo right now).

**Limitations for the paper** (from the plan): stop-point assumption, proxy validator, one-shot training vs iterative use, English only (non-English prompts get filler-only removal), evaluation cost, approximate token counts for closed models.
