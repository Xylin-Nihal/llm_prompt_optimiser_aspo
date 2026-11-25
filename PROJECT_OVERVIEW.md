# TokenWise / ASPO: Project Overview

*Last updated: 2026-11-20. Companion to `IMPLEMENTATION_PLAN.md`. The plan is the source of truth for decisions; this file explains the whole project, every formula and every decision in one place. If the two disagree, the plan wins and this file should be fixed.*

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [The core idea in plain words](#2-the-core-idea-in-plain-words)
3. [Key terms](#3-key-terms)
4. [Run-time workflow (one request)](#4-run-time-workflow-one-request)
5. [Offline workflow (training the policy)](#5-offline-workflow-training-the-policy)
6. [Modules](#6-modules)
7. [Protection rules (hard and soft spans)](#7-protection-rules-hard-and-soft-spans)
8. [Formulas, explained](#8-formulas-explained)
9. [A worked example](#9-a-worked-example)
10. [Design decisions and why](#10-design-decisions-and-why)
11. [Evaluation plan](#11-evaluation-plan)
12. [Related work and the gap](#12-related-work-and-the-gap)
13. [What is new, and what is not claimed](#13-what-is-new-and-what-is-not-claimed)
14. [Risks and limitations](#14-risks-and-limitations)
15. [Order of work and progress](#15-order-of-work-and-progress)

---

## 1. What the project is

**TokenWise** is a model-agnostic middleware. It sits between an application and an LLM API and shortens the prompt before it is sent, so each request uses fewer input tokens, costs less and returns faster.

**ASPO** (Adaptive Semantic Prompt Optimization) is the compression framework inside TokenWise. "Adaptive" means the amount of compression depends on the prompt itself (its task, complexity and structure), not on a fixed ratio.

Users adopt it with **one line of change**: point an OpenAI-compatible client at the TokenWise proxy.

```python
client = OpenAI(base_url="http://localhost:8000/v1")
```

**Two guarantees** (for users and for the paper):

| | Guarantee | What it means exactly |
|---|---|---|
| **G1** | Never worse than the original, as judged by the validator | If no compressed version passes the validator's checks, the original prompt is sent unchanged. This is a guarantee about the checks, not about task accuracy. |
| **G2** | Hard spans are never removed | Negations, constraints, format rules, protected spans in instructions and questions, code and placeholders survive byte-for-byte. The learned policy cannot remove them. |

---

## 2. The core idea in plain words

1. Split the prompt into **roles** (system, instruction, context, question, examples, code, format rules, chat history and so on).
2. Mark the parts that must never change (**hard spans**) and the parts that may only be removed whole (**soft spans**).
3. Remove **words** in small batches. A learned policy decides which words go first.
4. After each batch, run the **validator**. If it passes, keep the batch and continue. If it fails, put back the last passing version and stop.
5. Send the last passing version to the model.

The result is *the most compression that still passes the checks* for each prompt. That is the precise meaning of "maximum" in this project. The project does not claim the true maximum for every prompt; it claims the maximum the validator allows.

---

## 3. Key terms

| Term | Meaning |
|---|---|
| **P** | The original prompt. |
| **P′** | A compressed candidate of P. |
| **T_M(x)** | The number of tokens in text x under the **target model M's tokenizer** (for example, the tokenizer of GPT-4o-mini, Llama-3.1 or Qwen2.5). All savings are counted in T_M. |
| **Word** | A whitespace-separated unit with its attached punctuation (for example `"report,"`). This is the unit the loop removes, the same unit LLMLingua-2 drops. |
| **Subword** | A piece of a word in the policy encoder's own tokenizer (multilingual BERT). The policy scores subwords; a word's score is the mean of its subword scores. |
| **Removal unit** | Either a single word that does not overlap a hard span, or a whole soft span (removed all together or not at all). |
| **Segment** | A contiguous part of the prompt with one role (for example one paragraph of context). |
| **Role** | One of 11 labels: system, instruction, context, example, question, format, code, data, history, tool_output, other. |
| **Protected span** | A detected item inside a segment: code, URL, email, path, placeholder, quoted text, date, number, entity, negation, constraint, format rule (12 kinds). |
| **Hard span** | A protected span that may never be removed or altered (see §7). |
| **Soft span** | A protected span that may be removed whole, but never altered (see §7). |
| **e(x)** | The sentence embedding of text x, from `all-MiniLM-L6-v2`. |
| **cos(a, b)** | Cosine similarity between two embeddings, from −1 to 1 (in practice 0 to 1 for prompts). |
| **C** | The prompt's complexity score, from 0 (simple) to 1 (complex). Set by the analyzer. |
| **τ(C)** | The similarity threshold the compressed prompt must meet. Stricter for complex prompts. |
| **π_θ** | The learned removal policy, with parameters θ. |
| **NLI** | Natural Language Inference: a model that says whether one sentence entails another. Used to check that a shortened instruction still means the same thing. |

---

## 4. Run-time workflow (one request)

```
 Prompt (a string or OpenAI-style chat messages)
   │
   ▼
 [Gate]  T_M(P) < T_min (200 tokens)?  ── yes ──► send P unchanged ("skipped: short prompt")
   │ no
   ▼
 [M1 Segmenter]      split into 11 roles; mark hard and soft spans               ✅ built
   │
   ▼
 [M2 Analyzer]       task type + complexity C ∈ [0, 1]  →  sets τ(C)              ✅ built (v1)
   │
   ▼
 [M5.1 Normalize]    collapse whitespace, remove filler phrases, minify JSON
   │                 (no real content removed)
   ▼
 [M5.2 Removal loop] repeat:
   │                   1. score removable units (policy π_θ, or pilot scorer M3)
   │                   2. remove the k lowest-scored units
   │                   3. validate (M6)
   │                        pass → keep as the new best, continue
   │                        fail → restore the last passing text, stop
   │                 until the validator fails or nothing removable is left
   ▼
 [M6 Final check]    hard spans verbatim; every question and format segment present
   │                 (on failure: send the original P and log the reason)
   ▼
 [M7 Model Adapter]  LiteLLM → OpenAI / Anthropic / Gemini / Ollama
   │
   ▼
 [Report]            tokens before and after, TR, CR, $ saved, similarity,
                     stop reason, steps taken, latency, validator result per step
```

**Delivery surfaces (M8):** Python library `aspo.optimize(...)`, an OpenAI-compatible FastAPI proxy, a CLI (`python -m aspo optimize`) and a Streamlit dashboard.

---

## 5. Offline workflow (training the policy)

This is done once, before the policy is used at run time.

1. **Collect training prompts.** Use the training split of each dataset in §11 (never the test split). Segment them with M1 so every word knows its role and whether it is hard or soft.
2. **Start from a pretrained compressor.** Load the LLMLingua-2 multilingual BERT-base checkpoint (about 180M parameters). It already has a keep/drop head per token.
3. **Train with one-shot REINFORCE** (the same scheme as TACO-RL):
   - for each training prompt, the policy samples keep or drop for every removable unit in a single pass;
   - the resulting candidate P′ gets one reward (formula 12);
   - the policy is updated with the REINFORCE gradient (formula 14).
   Hard-span words are masked, so they are always kept.
4. **Tune hyperparameters on the 20% validation split only:** reward weights w_c and w_q, τ0 and τ1, T_min, δ_lex, the complexity weights, batch size k and validator frequency f.
5. **Evaluate on the test split** with the harness in `eval/`. Every number in the paper must come from `results/` produced by `eval/run_benchmark.py`.

**Compute.** Development, debugging and short runs on the laptop (RTX 4050 Laptop, 6 GB VRAM: enough for BERT-base with fp16 and small batches). Full training on Kaggle (free 16 GB GPU). Colab as a fallback. XLM-RoBERTa-large (the larger LLMLingua-2 model) is an optional extra run if time and GPU quota allow; it does not fit in 6 GB for training.

---

## 6. Modules

| Module | File | Status | What it does |
|---|---|---|---|
| **M0** Tokenizer layer | `aspo/tokenizers.py` | ✅ 2026-11-25 | `count_tokens(text, model)`: tiktoken for OpenAI models, Hugging Face tokenizers for open models, provider counting endpoint or a flagged approximation otherwise. `price_per_token(model)` from LiteLLM's cost table. LRU cache, because the loop counts tokens at every step. |
| **M1** Intake + Segmenter | `aspo/segmenter.py`, `aspo/protected.py`, `aspo/types.py`, `aspo/lexicon.py` | ✅ 2026-11-15 | Accepts text or chat messages. Assigns 11 roles. Detects 12 span kinds. Exact reconstruction (`to_text()`, `to_messages()`). Hard/soft tiers, removal units and intake checks (`aspo/intake.py`) added 2026-11-26. |
| **M2** Intent & Complexity Analyzer | `aspo/analyzer.py` | ✅ v1 2026-11-16 | 10 task types (qa, rag_qa, summarization, code, reasoning, extraction, classification, transformation, creative, chitchat) with probabilities, plus complexity C (formula 2). Length in T_M tokens done (with M0). To add: the prototype term (2b), a 400-prompt labelled evaluation set. |
| **M3** Pilot scorer | `aspo/scorer.py` | ⏳ | A cheap, non-learned scorer (IDF, or leave-one-out drift, formula 1). Proves the loop and validator work before the policy exists, and is a baseline in the ablations. |
| **M4** Learned policy | `aspo/policy.py`, `training/` | ⏳ | LLMLingua-2-style classifier fine-tuned with one-shot REINFORCE (formulas 12–14). Hard-span words forced to keep. Re-run at every loop step. |
| **M5** Normalize + removal loop | `aspo/optimizers.py` | ⏳ | M5.1 normalize (whitespace, filler phrases, JSON minify; never Python or YAML). M5.2 the removal loop (formulas 10–11). M5.3 optional lexical rewrite (formula 7), reported as a separate ablation. |
| **M6** Validator | `aspo/validator.py` | ⏳ next | Four checks (formula 9) inside the loop, and a final check after it. |
| **M7** Model adapter | `aspo/adapters.py` | ⏳ | LiteLLM interface to OpenAI, Anthropic, Gemini, Ollama. Passes streaming through. Tells M0 which tokenizer and price apply. |
| **M8** Delivery | `aspo/pipeline.py`, `server.py`, `dashboard.py`, `aspo/__main__.py` | ⏳ | Library, proxy (adds `X-TokenWise-*` headers), CLI, dashboard. Safeguards: overhead gate, fallback, logging off by default, `prefix_freeze` to keep provider prompt caching working. |
| Eval harness | `eval/` | ⏳ | `datasets.py`, `baselines.py`, `metrics.py`, `run_benchmark.py`. Writes `results/*.csv` and plots. |

**Optional dependencies** keep the core light: `[tok]` tiktoken, tokenizers · `[nlp]` sentence-transformers, spaCy, nltk, torch · `[serve]` litellm, fastapi, uvicorn, streamlit · `[eval]` datasets, rouge-score, bert-score, llmlingua.

---

## 7. Protection rules (hard and soft spans)

Protection has two tiers. The segmenter detects the spans and assigns each one a tier (built 2026-11-26). Spans in OTHER segments are hard.

### Hard spans: never removed, never altered

- NEGATION, CONSTRAINT and FORMAT spans, **in any role**.
- **Every detected protected span** inside an INSTRUCTION, QUESTION, FORMAT or SYSTEM segment, whatever its kind (for example, a number inside an instruction).
- CODE blocks.
- PLACEHOLDERs (such as `{name}` or `{{var}}`) anywhere.

Only the **spans** are hard, not the whole segment. The other words in an instruction, question, format or system segment can still be removed. The NLI check (N in formula 9) guards those removals, so a shortened instruction must still mean the same thing.

A word that overlaps a hard span counts as hard. The policy gives it a keep probability of 1, so it can never be chosen.

### Soft spans: removable whole, never altered

- NUMBER, DATE, ENTITY, URL, EMAIL, PATH and QUOTED spans in CONTEXT, DATA, HISTORY, TOOL_OUTPUT or EXAMPLE segments.
- Each soft span is an **atomic removal unit**: all its words are removed together, or none are. So "New York" or "3 March 2025" is either fully present or fully gone, never "New" or "3 2025".
- Soft-span retention (the share of soft spans that survive) is reported as a metric.

---

## 8. Formulas, explained

Every formula has the same number here, in the plan, in the code and (after revision) in the paper.

### (1) Pilot importance score (M3)

$$
I(u) = \alpha\,\big[1 - \cos\big(e(P),\, e(P \setminus u)\big)\big] + (1-\alpha)\,\cos\big(e(u),\, e(q)\big)
$$

- **u** is a unit (a sentence, or a word group); **P ∖ u** is the prompt with u removed; **q** is the question segment.
- The first term is **leave-one-out drift**: how much the whole prompt's meaning moves if u is deleted. Big drift means u matters.
- The second term is **relevance to the question**: how close u is to what is being asked.
- **α** balances the two. If there is no question, α = 1.
- Units with the **lowest** I(u) are removed first.

### (2) Complexity score (M2)

$$
C = \sum_{k} w_k\, f_k, \qquad \sum_k w_k = 1, \qquad k \in \{\text{task, reasoning, constraints, length, technical, structure}\}
$$

Six factors, each in [0, 1]:

| Factor | Formula | Meaning |
|---|---|---|
| f_task | $\sum_t p(t \mid P)\,\kappa_t$ | Expected difficulty of the task. κ_t is a fixed difficulty per task (reasoning 0.75, code 0.65, rag_qa 0.5, extraction 0.45, summarization 0.4, transformation 0.4, creative 0.35, classification 0.3, qa 0.3, chitchat 0.05). |
| f_reasoning | $\min\big(1,\ 0.25\,n_{\text{multistep}} + 0.15\,(n_{\text{subtasks}} - 1)\big)$ | Multi-step cues ("step by step", "first … then") and the number of sub-tasks. |
| f_constraints | $\min\big(1,\ n_{\text{hard instruction spans}} / 6\big)$ | How many constraint, negation and format rules the prompt has. Six or more counts as maximal. |
| f_length | $\min\big(1,\ \log(1+T_M(P)) / \log(1+T_{\text{ref}})\big)$ | Length on a log scale. In the current v1 code this uses words with T_ref = 4000 words; after M0 it switches to target-model tokens. |
| f_technical | $\min\big(1,\ 2\,\lvert\text{technical-span chars}\rvert / \lvert P\rvert\big)$ | Share of the prompt that is code, URLs, paths, numbers and similar. |
| f_structure | $\min\big(1,\ (n_{\text{distinct roles}} - 1) / 5\big)$ | How many different roles the prompt mixes. |

**Current weights:** task 0.25, reasoning 0.20, constraints 0.15, length 0.15, technical 0.15, structure 0.10. They will be tuned on the validation split.

### (2b) Task probabilities with a prototype term (optional, M2)

$$
p(t \mid P) = \operatorname{softmax}_t\big(\text{rule}_t + \beta\,\cos(e(P),\, \mu_t)\big)
$$

- **rule_t** is the current rule-based score for task t (role-weighted lexical cues plus structural features).
- **μ_t** is the average embedding of 5–10 hand-written prototype prompts for task t.
- **β** sets how much the embedding similarity counts. Without sentence-transformers installed, the analyzer falls back to rules only.

### (3) Validator threshold

$$
\tau(C) = \tau_0 + (\tau_1 - \tau_0)\,C, \qquad \tau_0 = 0.80,\ \tau_1 = 0.95
$$

A simple prompt (C = 0) must keep cosine similarity ≥ 0.80 to the original; the most complex prompt (C = 1) must keep ≥ 0.95. Complex prompts are compressed more cautiously.

### (4) Token reduction

$$
TR = \left(1 - \frac{T_M(P')}{T_M(P)}\right) \times 100\%
$$

The percentage of target-model tokens saved.

### (5) Compression ratio

$$
CR = \frac{T_M(P)}{T_M(P')}
$$

How many times shorter the prompt became. TR = 50% is the same as CR = 2.

### (6) Is it worthwhile? And cost saved (Eq. 9 in the paper draft)

$$
\text{worthwhile} \iff t_{\text{ASPO}} + t_{\text{LLM}}(P') < t_{\text{LLM}}(P), \qquad \Delta\text{Cost} = \big(T_M(P) - T_M(P')\big)\cdot p_{\text{in}}
$$

- Compression must save more model time than it costs to run. The prompt length where this first becomes true is the **break-even length**, reported in the evaluation.
- **p_in** is the model's price per input token.

### (7) Optional lexical rewrite (M5.3, not part of the core)

$$
w^{*} = \arg\min_{w' \in W(w)} T_M(w') \quad \text{s.t.}\quad T_M(w') < T_M(w),\ \ \text{sim}(w, w') \ge \delta_{\text{lex}}
$$

For a phrase w, choose the replacement w′ from a list W(w) (for example "in order to" → "to") that uses the fewest target tokens, but only if it is actually shorter under the target tokenizer and close enough in meaning (δ_lex). Reported as a separate ablation.

### (8) Gate

$$
T_M(P) < T_{\min}\ (=200) \ \Rightarrow\ \text{return } P \text{ unchanged}
$$

Short prompts are not worth compressing; the overhead would cost more than the saving.

### (9) Validator

$$
V(P') = 1 \iff H(P') \land Q(P') \land \big[\cos(e(P), e(P')) \ge \tau(C)\big] \land N(P')
$$

| Check | Passes when |
|---|---|
| **H** | Every hard span is present verbatim. |
| **Q** | Every QUESTION and FORMAT segment is still present (not removed entirely). |
| **S** (the cosine term) | The whole compressed prompt is at least τ(C) similar to the original. |
| **N** | Every instruction-bearing segment that was changed is still entailed in both directions by its original, with NLI probability ≥ 0.8 (`cross-encoder/nli-deberta-v3-small`). |

The **final check** after the loop repeats H and Q to catch bugs. If it fails, the original P is sent and the reason is logged.

### (10) Removal loop

$$
P_0 = \operatorname{Normalize}(P)
$$

For t = 1, 2, …:

$$
R_t = \text{removable units of } P_{t-1}, \qquad k_t = \max\!\big(1,\ \lceil 0.05\,\lvert R_t\rvert \rceil\big)
$$

$$
P_t = P_{t-1} \text{ with its } k_t \text{ lowest-scored units removed}, \qquad \text{score}(u) = \pi_\theta(\text{keep} \mid u) \ \text{or}\ I(u)
$$

Stop at the first t where V(P_t) = 0.

- A **removable unit** is a word that does not overlap a hard span, or a whole soft span.
- By default each step removes about **5%** of what is still removable (at least one unit). For the latency study, fixed k ∈ {1, 4, 16, 64} is also tested.
- The validator runs every **f** steps (f = 1 in the pilot; later f ∈ {1, 2, 4, 8} is measured).

### (11) Stop point

$$
t^{*} = \max\{\, t : V(P_s) = 1 \ \text{for all}\ s \le t \,\}, \qquad \text{output } P_{t^{*}}
$$

The output is the last version before the first failure. **Assumption:** once a step fails, later steps would also fail. This is not guaranteed (removing more could, in rare cases, pass again), so it is stated as a limitation, and the evaluation measures how often a later step would have passed.

### (12) Reward (training and tuning)

$$
R(P') = G(P')\cdot\left[\, w_c\,\frac{T_M(P) - T_M(P')}{T_M(P)} \;+\; w_q\,\cos\big(e(P), e(P')\big) \right]
$$

$$
G(P') = 1 \iff H(P') \land Q(P') \land \big[\cos(e(P), e(P')) \ge \tau(C)\big], \quad \text{otherwise } G = 0
$$

- The first term rewards **compression** (in target-model tokens). The second rewards **keeping the meaning**.
- **G is a hard gate**, not a weighted term: a candidate that breaks a check gets reward 0, no matter how short it is.
- G is the validator (9) **without the NLI check**, which is too slow to run on every training sample.
- With hard-span masking on, H and Q always hold, so the similarity threshold is what rejects candidates. H and Q matter in the "masking off" ablation.
- w_c + w_q = 1. Grid: w_c ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, tuned on the validation split only.
- No LLM API calls are needed during training.

### (13) Policy

$$
\pi_\theta(\text{keep} \mid w_i) = \frac{1}{\lvert w_i\rvert}\sum_{j \in w_i} \sigma\big(f_\theta(x)_j\big)
$$

- **f_θ(x)_j** is the encoder's output for subword j of the input x; **σ** is the sigmoid, giving a keep probability per subword.
- A word's keep probability is the **mean** over its subwords.
- A **soft span** is one unit: its keep probability is the mean over its words.
- Words overlapping a hard span are fixed to keep: π = 1.

### (14) REINFORCE, one-shot (as in TACO-RL)

$$
a_i \sim \operatorname{Bernoulli}\big(\pi_\theta(\text{keep} \mid u_i)\big)\ \text{for each removable unit } u_i, \qquad P' = \text{the units with } a_i = \text{keep}
$$

$$
\nabla_\theta J = \mathbb{E}\Big[\big(R(P') - b\big)\sum_i \nabla_\theta \log \pi_\theta(a_i \mid x)\Big], \qquad b = \text{mean reward over the batch}
$$

- For each training prompt the policy samples one keep/drop decision per removable unit, in a single pass, giving one candidate P′.
- If P′ scored **above** the batch average b, the decisions that produced it are made more likely; if below, less likely.
- Subtracting b (the **baseline**) does not change the expected gradient but reduces its variance, so training is more stable.
- **Limitation:** the policy is trained one-shot but used iteratively by formula (10) at run time. This is stated in the paper.

---

## 9. A worked example

Numbers are illustrative only; they are not results.

1. **Gate.** P has T_M(P) = 1000 tokens. 1000 ≥ 200, so compression goes ahead.
2. **Complexity.** Suppose the analyzer gives f_task 0.40, f_reasoning 0.25, f_constraints 0.33, f_length 0.50, f_technical 0.10, f_structure 0.40. Then
   C = 0.25·0.40 + 0.20·0.25 + 0.15·0.33 + 0.15·0.50 + 0.15·0.10 + 0.10·0.40 ≈ **0.33**.
3. **Threshold.** τ(0.33) = 0.80 + 0.15·0.33 ≈ **0.85**.
4. **Loop.** After normalizing, 300 units are removable. Step 1 removes k = ⌈0.05·300⌉ = **15** units, the validator passes, and so on. At some step the cosine drops below 0.85, so that step is undone and the loop stops.
5. **Result.** The last passing version has T_M(P′) = 400 tokens.
   TR = (1 − 400/1000)·100% = **60%**, CR = 1000/400 = **2.5**.
6. **Cost.** At an input price of $0.15 per million tokens, ΔCost = 600 × 0.15 / 1,000,000 = **$0.00009** per request ($90 per million requests).
7. **Reward (if this were a training sample).** cos = 0.90 ≥ 0.85, so G = 1. With w_c = 0.7, w_q = 0.3:
   R = 0.7·0.60 + 0.3·0.90 = 0.42 + 0.27 = **0.69**.
   If the same candidate had cos = 0.83, G = 0 and R = 0.

---

## 10. Design decisions and why

| Decision | Choice | Why |
|---|---|---|
| Segmentation | 11 roles (built) | Role-awareness is the main difference from earlier methods. |
| Protection | Two tiers, hard gate inside the policy | Simple, testable, and directly measurable (RQ4). |
| Removal unit | **Words** (subword scores averaged per word); soft spans as whole units | Same unit as LLMLingua-2, so comparison is fair. Output stays readable. Independent of the target tokenizer, which the RQ2 transfer test needs. |
| Savings unit | Target-model tokens T_M | Real cost depends on the target tokenizer (contribution C3). |
| Scorer | Learned policy (main), pilot scorer M3 first | The pilot proves the loop works and is a non-learned baseline. |
| Stop point | Set by the validator, not a fixed ratio | Removes hand-set budgets that reviewers would challenge (RQ1). |
| Training mode | **One-shot REINFORCE**, as in TACO-RL | Cheap and stable. Full-loop or per-step training is far more expensive, and per-step rewards would copy LLM-DCP. |
| Reward gate | H ∧ Q ∧ cos ≥ τ(C) | Training matches the run-time stop rule. NLI left out for cost. |
| Starting checkpoint | LLMLingua-2 multilingual BERT-base | Fits the 6 GB laptop GPU; XLM-R-large is an optional extra run. |
| Compute | Laptop for development, Kaggle for full training, Colab fallback | Kaggle's free 16 GB GPU and longer sessions suit full training runs. |
| Batch size k | About 5% of removable units per step; sweep {1, 4, 16, 64} | Larger k is faster but may compress less; the sweep is the latency report. |
| Validator frequency f | 1 in the pilot; sweep {1, 2, 4, 8} later | Correctness first, then measure the speed/accuracy trade-off. |
| Lexical rewrite | Optional extra, not in the core loop | Small savings and a risk of changing meaning. |
| Intent detection | Rules (built) + optional prototypes | No labelled training data yet; a fine-tuned classifier is out of scope. |
| Serving | LiteLLM + FastAPI proxy + Streamlit | The proxy makes the "middleware" claim real. |
| Cache-awareness | `prefix_freeze` option | Cheap, and keeps provider prompt caching working. |
| Tuning | All hyperparameters on the 20% validation split only | Never on the test split. |

---

## 11. Evaluation plan

### Research questions

| RQ | Question |
|---|---|
| **RQ1** | At matched quality, does the validator-controlled loop achieve more compression than a fixed-ratio stop? |
| **RQ2** | Does target-tokenizer accounting change the reported savings, and does a policy trained for one tokenizer transfer to another? |
| **RQ3** | How much does the validator reduce quality failures, and what does it cost in latency and savings? (Validator precision and recall against human labels; failure rate with and without the validator; validator-frequency sweep.) |
| **RQ4** | Does hard-span masking reduce constraint violations compared with baselines that do not protect constraints? (Prompt level, deterministic; output level, IFEval checkers.) |
| **RQ5** (optional) | Does `prefix_freeze` lower effective cost when provider caching is on? |

### Datasets (200 test items each, plus a 20% validation split)

| Task family | Dataset | Quality metric |
|---|---|---|
| Summarization | Arxiv-March23 | ROUGE-L, BERTScore |
| Conversation | ShareGPT | ROUGE-L, BERTScore |
| Math reasoning | GSM8K (few-shot chain-of-thought prompt) | Exact match |
| In-context learning | BBH (6 tasks) | Exact match |
| RAG / long context | LongBench (HotpotQA, MultiFieldQA) | F1 |
| Constraint stress | 150 items from the sets above, each with one IFEval-style verifiable instruction appended | Constraint-violation rate |

### Target models (three tokenizers)

One low-cost OpenAI model, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (via Ollama). Temperature 0; all responses cached.

### Baselines

No compression · random word removal (same ratio as ASPO) · truncation · Selective Context · LLMLingua · LongLLMLingua (RAG set only) · LLMLingua-2 · LLM-DCP · TACO-RL · the pilot loop (M3) without the learned policy.

### Metrics

TR, CR, $ saved; latency (t_ASPO, t_LLM, total, break-even length from formula 6); task quality; cos(e(P), e(P′)); validator acceptance and fallback rates; validator precision and recall against human labels; hard-span retention (must be 100%); soft-span retention; constraint-violation rate (prompt and output level); stop-reason distribution; steps taken.

### Ablations

- Fixed-ratio stop vs validator stop
- Learned policy vs pilot scorer M3
- Hard-span masking on vs off
- Validator on vs off; cosine only vs cosine + NLI
- Validator frequency f ∈ {1, 2, 4, 8}
- Batch size k ∈ {1, 4, 16, 64} and the 5% default
- Analyzer rules only vs rules + prototypes
- Optional: with vs without lexical rewrite

### Statistics

Three seeds for any sampling. 95% bootstrap confidence intervals. Paired bootstrap (ROUGE, F1) or McNemar's test (exact match, constraint violations) against the best baseline. Cost–quality Pareto curves.

**Integrity rule:** no number goes into the paper unless `eval/run_benchmark.py` produced it and it is saved in `results/`.

---

## 12. Related work and the gap

*The first four entries were checked through search summaries, not full-text reads. Check every entry against the paper before citing it.*

| Method | What it does | What it lacks for this project |
|---|---|---|
| **LLMLingua** (Jiang et al., EMNLP 2023, arXiv 2310.05736) | Budget controller over instruction, demonstrations and question; small causal LM scores perplexity; iterative token pruning. | Needs manual prompt splitting; no runtime validator. |
| **LongLLMLingua** (arXiv 2310.06839) | Question-aware coarse compression of documents; dynamic budgets; reorders documents. | Question-dependent output limits caching; no validator. |
| **LLMLingua-2** (Pan et al., Findings of ACL 2024, arXiv 2403.12968) | BERT-style keep/drop classifier trained on GPT-4-distilled labels; task-agnostic; fast. | No constraint protection inside the policy; fixed ratio; no runtime check. |
| **LLM-DCP** (arXiv 2504.11004) | Compression as a Markov Decision Process; policy removes tokens step by step; reward with ratio bounds, KL divergence via a small aligned model, BERTScore. | No hard constraint gate; no runtime validator deciding the stop point. |
| **TACO-RL** (arXiv 2409.13035) | Fine-tunes the LLMLingua-2 encoder with REINFORCE and task-specific rewards; single pass. | No constraint protection; no runtime validator. |
| **PCRL** (arXiv 2308.08758) | Policy network edits prompts directly. | Not verified in detail. |
| **PIS** (arXiv 2504.16574) | Attention-based saliency with a small RL network, plus a sentence stage. | Not verified in detail. |
| **Selective Context** (Li et al., 2023) | Removes low self-information content using a causal LM. | Baseline; not verified in detail. |
| **IFEval** (Zhou et al., 2023) | Verifiable instruction checkers. | Used for the constraint-stress set. |

**The gap:** no surveyed method combines automatic role detection, a hard constraint gate inside the policy, a runtime validator that sets the stop point, and target-tokenizer accounting.

---

## 13. What is new, and what is not claimed

### Contributions

| # | Contribution | Evidence |
|---|---|---|
| **C1** | Role-aware segmentation with two-tier protection | Segmenter accuracy; constraint-violation rate (RQ4); "no protection" ablation |
| **C2** | Validator-controlled learned removal policy | RQ1; quality–ratio curves |
| **C3** | Target-tokenizer accounting | Savings per tokenizer; optional lexical-rewrite ablation |
| **C4** | Run-time validate-and-stop loop | RQ3; validator precision and recall; validator-frequency cost |
| **C5** | Deployable middleware | Latency and break-even (formula 6); batch-size sweep; case study |

### Claims the paper must **not** make

- That the removal policy itself is new. It is close to LLM-DCP and TACO-RL.
- That the validator guarantees task accuracy. It guarantees its checks, which are proxies.
- That the loop reaches the true maximum compression for every prompt. Only the validator-limited maximum is claimed.

---

## 14. Risks and limitations

| # | Risk | Response |
|---|---|---|
| R1 | The policy is close to LLM-DCP and TACO-RL. | Present it as a component; let the validator and protection results carry the claim. |
| R2 | Hard-span masking may lower compression compared with token-level baselines. | Report quality at matched tokens, not raw compression alone. |
| R3 | The validator uses proxy checks (cosine, NLI). | Report precision and recall against human labels. |
| R4 | The stop-point assumption (formula 11) may not hold. | State it as a limitation; measure how often later steps would have passed. |
| R5 | Hand-set weights (complexity, τ0, τ1) can be challenged. | Tune only on the validation split; report the grid. |
| R6 | Re-running the policy every step adds latency. | Report batch-size and validator-frequency trade-offs and the break-even length. |
| R7 | The 400-prompt analyzer labels need agreement. | Two annotators; report Cohen's κ. |
| R8 | The policy is trained one-shot but run iteratively. | State it as a limitation; compare with the pilot scorer, which has no such mismatch. |

**Limitations to state in the paper:** stop-point assumption, proxy validator, one-shot training vs iterative use, English only (non-English prompts get filler-only removal), evaluation cost.

**Go / no-go checkpoint (weeks 3–4):** compare the pilot loop with LLMLingua-2 on one dataset. If ASPO does not match LLMLingua-2 on quality at equal tokens, the paper leads with C1, C4 and RQ4, and presents compression as a trade-off.

---

## 15. Order of work and progress

**Order of work** (the loop and validator are built before the policy, so each step is testable):

1. M0 tokenizer layer — counts match the official tokenizers.
2. M6 validator — one unit test per check.
3. M5 normalize and removal loop with the pilot scorer M3 — stop-and-restore works on example prompts.
4. Pilot comparison against LLMLingua-2 on one dataset — go / no-go recorded.
5. M4 learned policy with one-shot REINFORCE — beats the pilot scorer on the validation split.
6. M7 adapter, M8 pipeline, proxy, CLI, dashboard — an OpenAI SDK pointed at the proxy works end to end (mid-term demo).
7. Full evaluation — baselines, main runs, ablations, sweeps, statistics.
8. Paper — filled only with results from `results/`.

**Progress (2026-11-20):**

- [x] M1 segmenter, protected spans, types, lexicon, CLI
- [x] M2 analyzer v1, CLI
- [x] 81 tests passing (48 segmenter and protected-span, 33 analyzer)
- [x] Plan decisions: formula numbering, word-level removal, one-shot REINFORCE, reward gate, soft spans as atomic units, hard-tier reading, checkpoint and compute
- [ ] M0 tokenizer layer ← **next**
- [ ] M6 validator
- [ ] M5 normalize + removal loop with M3
- [ ] Pilot comparison vs LLMLingua-2 (go / no-go)
- [ ] M4 learned policy
- [ ] M7, M8
- [ ] Evaluation harness, baselines, ablations, statistics
- [ ] Paper
