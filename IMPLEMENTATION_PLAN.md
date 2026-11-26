# TokenWise / ASPO: Unified Implementation & Research Plan

**TokenWise** is a model-agnostic middleware that shrinks LLM prompts before they reach the model. **ASPO** (Adaptive Semantic Prompt Optimization) is the framework inside it. The core is a **learned word-removal policy** that runs inside a **validate-and-stop loop**: words are removed in batches until the validator says the next step would break the prompt, and the last passing version is sent.

This plan is the single source of truth for the code and the paper. Where the paper draft and the plan disagree, the decision and the reason are recorded in §3.

*Last updated: 2026-11-27 · Status: M0 Tokenizer ✅ · M1 Segmenter ✅ (tiers + intake) · M2 Analyzer (v1) ✅ · M6 Validator ✅ · removal loop, pilot scorer, policy ⏳*

---

## 0. Change log

- **2026-11-20:** Formula references fixed to match §6. Removal unit fixed to **words** (as in LLMLingua-2); savings still counted in T_M. Policy training fixed to **one-shot REINFORCE** (as in TACO-RL), with the reward gated by H ∧ Q ∧ cos ≥ τ(C). Starting checkpoint and compute chosen (§5 M4, §7).
- **2026-11-18 (revision):** The importance scorer (M3) and budget allocator (M4) are replaced by a **learned token-removal policy** inside a **validate-and-stop loop**. Constraint protection becomes a hard gate, not a reward term. Lexical rewriting (M5.4) and the tokenizer-aware reward are removed from the core loop and kept as optional extras. The learned policy moves from future work (§10) into the core. Earlier decisions are kept where still valid.

---

## 1. Goal

Reduce the input tokens (and therefore cost and latency) of real LLM requests, as far as possible while every hard constraint and the validator's checks still pass. Users should be able to adopt it with **one line of change**: point an OpenAI-compatible client at the TokenWise proxy.

Two non-negotiable guarantees, for users and for the paper:
1. **Never worse than the original, by the validator's checks.** If no compressed version passes validation, the original prompt is sent unchanged. The guarantee is about the checks, not about task accuracy.
2. **Hard spans are never removed.** Negations, constraints, format rules, questions, code and placeholders survive byte-for-byte. The policy cannot remove them.

---

## 2. Contributions (what the paper claims)

| # | Contribution | Evidence in the paper |
|---|---|---|
| **C1** | **Role-aware segmentation with two-tier protection.** Arbitrary prompts and chat transcripts are split automatically into 11 roles. Instruction-type spans are *hard-protected*. Content spans are *soft-protected*: never altered, but a span may be removed as a whole. | Segmenter accuracy on hand-labelled prompts; constraint-violation rate (RQ4); "no protection" ablation |
| **C2** | **Validator-controlled learned removal policy.** A word-level policy (the same unit as LLMLingua-2) removes words in batches. The stopping point is set by the validator, so each prompt gets the most compression that still passes the checks, instead of a fixed or hand-set ratio. | RQ1: maximal-under-validator vs fixed ratio at matched quality; quality–ratio curves |
| **C3** | **Target-tokenizer accounting.** All savings are reported in the target model's tokens and cost, not generic counts. Tokenizer-aware lexical rewriting is an optional extra, reported separately. | Savings per tokenizer (three tokenizers); optional lexical-rewrite ablation |
| **C4** | **Run-time validate-and-stop loop.** A complexity-dependent similarity threshold τ(C), NLI equivalence for rewritten instructions, and hard-span and question/format checks decide when to stop. The last passing version is restored. | RQ3: failure rate with vs without the validator; validator precision and recall against human labels; validator-frequency cost |
| **C5** | **Deployable middleware.** OpenAI-compatible proxy, Python library, CLI and dashboard, with a per-request report (token reduction, compression ratio, cost saved, similarity, stop reason). | Latency and break-even analysis (formula (6)); batch-size sweep; case study |

**Novelty statement (honest version).** LLMLingua-2 and LLM-DCP learn which tokens to drop, but neither protects constraints at the policy level, neither uses a runtime validator to decide how far to compress, and neither optimises for the target tokenizer. LLMLingua and LongLLMLingua need the user to split the prompt into parts by hand, and they have no runtime check. ASPO combines automatic role detection, a hard constraint gate inside the policy, a validator that decides the stopping point, and target-tokenizer accounting. No surveyed method does all of these together. The learned policy itself is close to LLM-DCP and TACO-RL, so the paper must not claim the policy as new.

---

## 3. What we took from where

| Topic | Paper draft | Original plan | **Decision** | Why |
|---|---|---|---|---|
| Segmentation | Sentences only | 11 roles + protected spans | **11 roles** (built) | Role-awareness is the main differentiator and works |
| Protection | Any span with a protected item is undeletable | Spans kept, text around them compressible | **Two tiers, hard gate** | Simple and testable; the policy masks hard spans |
| Intent detection | Prototype embeddings + rules | Embeddings + LR, then DeBERTa | **Rules (built) + prototypes** | No labelled training data yet; DeBERTa is out of scope |
| Complexity | Eq. 2: 4 terms | Not formalised | **Six-factor formula** (§6) | Built and tested |
| Importance | Leave-one-out drift per segment | Token self-info + relevance + classifier | **Learned word-level policy** (main); LOO + relevance kept as the pilot scorer | The paper claims the learned policy; the pilot scorer proves the loop works first |
| Budget | One global r(C) | Per-segment rule table → bandit | **Validator decides the stop point**; per-segment budget removed | Removes the hand-set table that reviewers would challenge |
| Optimizers | Dedup, filler, deletion, lexical | Plus restructuring, MinHash, LLM rewrite | **Normalize (filler, whitespace, JSON minify) → removal loop**; lexical rewrite optional | Lexical rewrite gives small savings and risks meaning |
| Validator | cos ≥ τ(C), restore, fallback | Fixed cos + NLI + span checks | **Merged, run inside the loop** | Each step must be checked, not only the final output |
| Serving | LangChain + Streamlit | LiteLLM + FastAPI + Streamlit | **LiteLLM + FastAPI proxy + Streamlit** | The proxy makes the "middleware" claim real |
| Datasets | Arxiv, ShareGPT, GSM8K, BBH | Many | **Paper's 4 + LongBench + constraint-stress set** | Matches LLM-DCP for direct comparison |
| Baselines | SC, LLMLingua, LLMLingua-2, LLM-DCP | Union | **Union + TACO-RL** | TACO-RL is the closest RL baseline |
| Statistics | Point estimates | CIs, tests | **CIs and paired tests** | Needed for acceptance |
| Cache-awareness | Future work | Strongly recommended extra | **`prefix_freeze` option**, evaluated only if time allows | Cheap; useful to users |

---

## 4. Architecture

```
 App / user ──► [Proxy · Library · CLI · Dashboard]                          (M8)
                       │
                       ▼
 [M0 Tokenizer layer: T_M(·), prices]  ◄── used by every stage below         ✅
                       │
 [Gate] T_M(P) < T_min ? ──yes──► send P unchanged
                       │ no
 [M1 Intake + Segmenter] roles, hard/soft spans                              ✅
 [M2 Intent & Complexity Analyzer] task, C ∈ [0,1]                           ✅ v1
 [M5.1 Normalize] whitespace, filler, JSON minify (no content removed)
                       │
 [M5.2 Removal loop]  ◄── policy π_θ (learned, RL-trained) or pilot scorer M3
        repeat: score removable words → remove a batch of k words
                → [M6 Validator]  pass: keep and continue
                                  fail: restore last passing text, stop
                       │
 [M6 Final check] hard spans verbatim, question and format present
                       │
 [M7 Model Adapter] LiteLLM → OpenAI / Anthropic / Gemini / Ollama …
                       │
 [Report] tokens before/after, TR, CR, $ saved, stop reason, steps, latency
```

Offline, the **policy training** loop (§7) produces π_θ. Online, π_θ is used only inside the removal loop.

---

## 5. Modules

### M0 — Tokenizer layer `aspo/tokenizers.py` ✅
- `count_tokens(text, model) -> int`: `tiktoken` for OpenAI models; Hugging Face `AutoTokenizer` for open models (Llama-3.1, Qwen2.5). For closed models without a public tokenizer, use the provider's counting endpoint if one exists; otherwise a documented approximation, flagged in the report.
- `price_per_token(model)`: input price from LiteLLM's cost table.
- LRU cache on encodings, since the removal loop calls it at every step.
- Tests: counts match the official tokenizers on fixed strings.
- Knock-on: the analyzer's length factor switches from words to `T_M` tokens.

### M1 — Intake + Segmenter `aspo/segmenter.py`, `aspo/protected.py` ✅
Built and tested (§12). Two-tier mapping and intake (empty + context-window) done 2026-11-26. Still to add:
- Simple language check. Non-English prompts use a "safe" preset (removal limited to filler only) until the multilingual extension.
- **Two-tier protection** (span kinds are already detected; the tier mapping is new):
  - **Hard:** NEGATION, CONSTRAINT and FORMAT spans in any role; every detected protected span inside INSTRUCTION, QUESTION, FORMAT or SYSTEM segments, whatever its kind; CODE blocks; PLACEHOLDERs anywhere. These are never removed or altered. The policy masks them out. Only the spans are hard, not the whole segment: the other words in an INSTRUCTION, QUESTION, FORMAT or SYSTEM segment can still be removed, and the NLI check (N in formula (9)) guards those removals. The Q check only requires that each QUESTION and FORMAT segment is not removed entirely.
  - **Soft:** NUMBER, DATE, ENTITY, URL, EMAIL, PATH and QUOTED spans in CONTEXT, DATA, HISTORY, TOOL_OUTPUT or EXAMPLE segments. Each soft span is an **atomic unit**: all of its words are removed together or none are, so it is always either verbatim or entirely gone, never altered. Soft-span retention is reported as a metric.

### M2 — Intent & Complexity Analyzer `aspo/analyzer.py` ✅ v1
Built: 10 task types, role-weighted lexical cues plus structural features, a softmax, a six-factor complexity score, explanation signals and a feature vector (§12). To add:
- ~~Length factor in target-model tokens (after M0).~~ Done 2026-11-25 (`AnalyzerConfig.target_model`).
- **Prototype-embedding scorer:** 5–10 prototype prompts per task, embedded with `all-MiniLM-L6-v2`. The cosine to each task centroid is added to the rule logits (formula (2b) in §6). Optional dependency; falls back to rules only.
- **Analyzer evaluation set:** 400 prompts from LMSYS-Chat-1M or ShareGPT, labelled independently by two people. Report Cohen's κ and macro-F1 for rules-only vs rules + prototypes. This is a paper table.
- Role in the new design: the analyzer supplies the task type and C, which set the validator's strictness through τ(C).

### M3 — Pilot removal scorer `aspo/scorer.py` ⏳ (built before the policy)
- A simple, cheap scorer for the removal loop. It is used to test the loop and as the **pilot baseline**.
- Word score: inverse document frequency over the prompt, or leave-one-out drift at sentence level (formula (1)). The lowest-scoring words are removed first.
- Purpose: prove that the loop and the validator work before the policy is trained, and provide a non-learned baseline for the ablations.

### M4 — Learned removal policy `aspo/policy.py`, `training/` ⏳ (after the pilot)
- **Unit of removal: words.** A word is a whitespace-separated unit with its attached punctuation, the same unit LLMLingua-2 drops. The encoder works on subwords, and a word's score is the mean of its subword scores. A word that overlaps a hard span counts as hard. The words of a soft span form one removal unit, scored by the mean over its words and removed all together or not at all. Savings are still counted in target-model tokens T_M, so the policy does not depend on one target tokenizer (needed for the transfer test in RQ2).
- Architecture: an LLMLingua-2-style token classifier, meaning a transformer encoder with a per-token keep/drop head.
- **Starting checkpoint:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (multilingual BERT-base, about 180M parameters). The XLM-RoBERTa-large checkpoint (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`) is an optional extra run if time and GPU quota allow. Confirm both against the LLMLingua-2 paper and model cards.
- Input: the current (partly compressed) prompt, with a role tag per word and a hard-span mask.
- Output: π_θ(keep | word) for each removable word.
- **Hard-span masking:** words that overlap hard spans are forced to keep. The policy cannot remove them.
- **Training: one-shot REINFORCE, as in TACO-RL** (formula (14)). For each training prompt, the policy samples keep/drop for every removable word in a single pass. The one resulting candidate gets one reward (formula (12)), and the policy is updated with a mean-reward baseline. The iterative loop is used only at inference. Limitation to state in the paper: the policy is trained one-shot but run iteratively.
- **Reward gate in training:** H ∧ Q ∧ cos(e(P), e(P′)) ≥ τ(C), which is the validator without NLI. NLI is left out of training to keep it cheap. No LLM API calls are made during training.
- Inference: the π_θ scores pick the k lowest-scoring removable words at each step. The classifier is re-run on the current prompt at each step, so the iterative cost is linear in the number of steps.
- **Compute:** development, debugging and short runs happen on the laptop (RTX 4050 Laptop, 6 GB VRAM: enough for mBERT-base with fp16 and small batches, plus MiniLM for the reward). Full training runs on Kaggle (free 16 GB T4 or P100; check the current weekly GPU quota and session limit). Colab free is the fallback. Save checkpoints regularly so a session limit does not lose work.

### M5 — Normalize and removal loop `aspo/optimizers.py` ⏳
1. **Normalize** (no content removed): collapse repeated whitespace and blank lines; remove a curated list of filler and politeness phrases ("could you please kindly…", "I was wondering if…") outside protected spans and code; JSON minify (never Python or YAML, where whitespace matters).
2. **Removal loop** (formulas (10) and (11)): repeat, starting from the normalized prompt, until the validator fails or no removable word is left:
   - score the removable words with the active scorer (M3 pilot or M4 policy);
   - remove the k lowest-scored removable words;
   - validate every f steps (f = 1 in the pilot);
   - on failure, restore the last passing text and stop.
3. **Optional lexical rewrite** (an extra, not a core claim): verbose-to-concise phrase substitutions, accepted only if they save tokens under the target tokenizer and pass NLI. Reported as a separate ablation.

### M6 — Validator `aspo/validator.py` ✅ (2026-11-27)
A candidate P′ passes (formula (9)) only if all of these hold:
1. every hard span is present verbatim;
2. every QUESTION and FORMAT segment is still present;
3. cos(e(P), e(P′)) ≥ τ(C);
4. every instruction-bearing segment changed by a removal or rewrite passes bidirectional NLI entailment against its original (`cross-encoder/nli-deberta-v3-small`, p ≥ 0.05, chosen by `eval/nli_probe.py`).

Final check (after the loop): repeat checks 1–2 on the output, to catch any bug in the loop. If it fails, send the original P and log the reason.

Validator frequency f: 1 in the pilot. Later, f is measured (for example 1, 2, 4, 8), and the accuracy loss and cost of each setting are reported.

**Build plan (agreed 2026-11-27)**

- **Input:** the original `SegmentedPrompt` plus `{segment_id: new_text}` (the format `to_text()` / `to_messages()` already take). The C value comes from M2.
- **Check order:** H → Q → S → N, cheapest first. Short-circuit at the first failure inside the loop; `short_circuit=False` computes every value for the evaluation.
- **H:** the merged hard intervals of each segment appear verbatim, in their original order, in that segment's new text.
- **Q:** each QUESTION and FORMAT segment still has non-blank text.
- **S:** cos(e(P), e(P′)) ≥ τ(C) = 0.80 + 0.15·C. **Decision: chunked mean.** e(P) = the length-weighted mean of MiniLM embeddings over chunks below 256 word pieces, taken segment by segment, because all-MiniLM-L6-v2 truncates at 256. Embeddings are cached by chunk text, so unchanged segments are free.
- **N:** **Decision: built now, on by default, with a switch.** Bidirectional entailment, p ≥ 0.05 (was 0.8; see the 2026-11-27 NLI decision below), with `cross-encoder/nli-deberta-v3-small`, on every changed segment whose role is INSTRUCTION, QUESTION, FORMAT, SYSTEM or **OTHER** (decision: OTHER included, matching its hard tier). Cached by (premise, hypothesis).
- **Final check:** H and Q on the rebuilt output text or messages. If it fails, the original is returned with the reason.
- **Output:** `ValidationResult`: `passed`, the first failed check (used as the stop reason), and for each check its value, threshold, details and time. Also reports soft-span retention as a metric.
- **Ablation hooks:** turn each check on or off; set τ0, τ1 and the NLI threshold; `gate()` = H ∧ Q ∧ S (the M4 reward gate G).
- **Models:** the embedder and the NLI model sit behind small interfaces. Unit tests use fake deterministic models; integration tests use the real ones (skipped if they are not installed). GPU if available. **Decision: CUDA build of PyTorch** in `.venv`, which M4 needs anyway.
- **Files:** `aspo/validator.py`, `tests/test_validator.py`. sentence-transformers goes in the `[nlp]` extra.
- **Risks to measure:** NLI may score shortened text too low and reject too often (measure on the examples). DeBERTa truncates at 512 tokens, so very long instruction segments are only partly checked.

### M7 — Model Adapter `aspo/adapters.py` ⏳
LiteLLM, giving one interface for OpenAI, Anthropic, Gemini and Ollama. Passes streaming through unchanged. Tells M0 which tokenizer and prices apply to the target model.

### M8 — Delivery surfaces ⏳
- **Library:** `aspo.optimize(prompt, model, preset="balanced") -> (optimized, report)` in `aspo/pipeline.py`.
- **OpenAI-compatible proxy** `server.py` (FastAPI): `POST /v1/chat/completions`. Optimizes the final user message (and history or tool outputs). Adds `X-TokenWise-*` headers with savings and stop reason. Usage: `OpenAI(base_url="http://localhost:8000/v1")`.
- **CLI:** `python -m aspo optimize file.txt --model gpt-4o-mini`.
- **Dashboard** `dashboard.py` (Streamlit): side-by-side diff with removed text highlighted, per-segment roles, validator results per step, stop reason, a preset slider, and token and cost savings.
- **Safeguards:** overhead gate, fallback guarantee, no logging by default (logs are opt-in and stored locally), and a `prefix_freeze` option that leaves the system prompt and tool definitions untouched so provider prompt caching keeps working.

---

## 6. Formal model (keep the paper and the code identical)

```
        e(·) = all-MiniLM-L6-v2 sentence embedding; T_M(·) = token count under target model M; a "word" is a whitespace unit with attached punctuation
(1)   I(u)     = α·[1 − cos(e(P), e(P∖u))] + (1−α)·cos(e(u), e(q))        pilot scorer (M3)
(2)   C        = Σ_k w_k·f_k ,  Σ w_k = 1,  k ∈ {task, reasoning, constraints, length, technical, structure}
        f_length      = min(1, log(1+T_M(P)) / log(1+T_ref))
        f_task        = Σ_t p(t|P)·κ_t
        f_constraints = min(1, n_hard_instruction_spans / 6)
        f_technical   = min(1, 2·|tech-span chars| / |P|)
        f_reasoning   = min(1, 0.25·n_multistep_cues + 0.15·(n_subtasks − 1))
        f_structure   = min(1, (n_distinct_roles − 1) / 5)
        current weights: 0.25, 0.20, 0.15, 0.15, 0.15, 0.10 (tuned on the validation split)
(2b)  p(t|P)   = softmax(rule_t + β·cos(e(P), μ_t))                        prototype term (M2, optional)
(3)   τ(C)     = τ0 + (τ1 − τ0)·C ,   τ0 = 0.80,  τ1 = 0.95
(4)   TR       = (1 − T_M(P′)/T_M(P))·100%
(5)   CR       = T_M(P) / T_M(P′)
(6)   worthwhile iff  t_ASPO + t_LLM(P′) < t_LLM(P);   ΔCost = (T_M(P) − T_M(P′))·p_in        (Eq. 9 in the paper draft)
(7)   w*       = argmin_{w′∈W(w)} T_M(w′)  s.t.  T_M(w′) < T_M(w),  sim(w, w′) ≥ δ_lex       optional lexical rewrite
(8)   Gate:    T_M(P) < T_min (default 200) → return P unchanged
(9)   V(P′)    = 1  iff  H(P′) ∧ Q(P′) ∧ [cos(e(P), e(P′)) ≥ τ(C)] ∧ N(P′)
        H: every hard span present verbatim
        Q: every QUESTION and FORMAT segment present
        N: bidirectional NLI entailment, p ≥ 0.05, for every changed instruction segment
(10)  Removal loop:  P_0 = Normalize(P);  for t = 1, 2, …
        R_t   = removable units of P_{t−1}       (words not overlapping a hard span; each soft span = one unit)
        k_t   = max(1, ⌈0.05·|R_t|⌉) units        (default; swept over fixed k ∈ {1, 4, 16, 64})
        P_t   = P_{t−1} with the k_t lowest-scored units removed
        score(w) = π_θ(keep | w)  (policy, formula (13))  or  I(w)  (pilot)
        stop at the first t where V(P_t) = 0
(11)  Stop point:  t* = max{ t : V(P_s) = 1 for all s ≤ t };   output P_{t*}
(12)  Reward (per candidate P′, used in training and tuning):
        R(P′) = G(P′) · [ w_c · (T_M(P) − T_M(P′)) / T_M(P)  +  w_q · cos(e(P), e(P′)) ]
        G(P′) = 1  iff  H(P′) ∧ Q(P′) ∧ [cos(e(P), e(P′)) ≥ τ(C)]   (the validator (9) without N); otherwise 0
        With hard-span masking on, H and Q always hold, so the τ(C) term is what rejects candidates;
        H and Q matter in the "masking off" ablation.
        w_c + w_q = 1;  grid over w_c ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, tuned on the validation split
(13)  Policy:  π_θ(keep | w_i) = mean_{j ∈ w_i} σ(f_θ(x)_j)   (mean over the subwords j of word w_i)
        words overlapping a hard span are fixed to keep (π = 1)
        a soft span is one unit: π_θ(keep | span) = mean over its words; in (14) one action a is sampled per soft span
(14)  REINFORCE (one-shot, as in TACO-RL):
        a_i ~ Bernoulli(π_θ(keep | w_i)) for every removable word i;   P′ = the words with a_i = keep
        ∇_θ J = E[ (R(P′) − b) · Σ_i ∇_θ log π_θ(a_i | x) ],  b = mean reward over the batch
        The policy is trained one-shot and run iteratively by (10) at inference (a stated limitation).
```

Note on formula (11): the stop point assumes that once a step fails, later steps would also fail. This is not guaranteed. State it as a limitation in the paper. An alternative is to keep going after a failure and retain the best passing version, at higher cost.

---

## 7. Training and tuning (decisions)

| Decision | Choice |
|---|---|
| Granularity | word-level policy, as in LLMLingua-2: subword scores averaged per word, whole words removed, savings counted in T_M |
| Batch size k | start at about 5% of removable words per step; sweep fixed values {1, 4, 16, 64} for the latency-versus-compression curve |
| Validator frequency f | every step in the pilot; once the pilot is stable, sweep f and measure accuracy loss and cost for each setting |
| Constraints | hard gate (formula (12), G), not a weighted reward term |
| Training mode | one-shot REINFORCE, as in TACO-RL: one sampled candidate per prompt, one reward; the iterative loop is used only at inference |
| Reward gate | H ∧ Q ∧ cos ≥ τ(C), i.e. the validator without NLI (NLI too costly in training) |
| Starting checkpoint | LLMLingua-2 multilingual BERT-base; XLM-RoBERTa-large as an optional extra run |
| Compute | laptop RTX 4050 (6 GB VRAM) for development and short runs; Kaggle (16 GB GPU) for full training; Colab as fallback |
| Reward weights | w_c and w_q only; tuned on the validation split with the grid above; never on the test split |
| Policy training data | training split of the datasets in §8, role-labelled by M1 |
| Hyperparameters | τ0, τ1, T_min, δ_lex, the complexity weights, k, f and w_c are all tuned on the 20% validation split |

Order of work (so the policy is not trained before the loop is correct):
1. Build M0 and the validator, with unit tests for each check.
2. Build the removal loop with the pilot scorer M3. Check that stop-and-restore works on the example prompts.
3. Run the pilot comparison against LLMLingua-2 on one dataset (go / no-go decision).
4. Train the policy M4 with REINFORCE, and compare it with the pilot scorer.

---

## 8. Evaluation (the core of the paper)

**Research questions**
- **RQ1:** At matched quality, does the validator-controlled loop achieve more compression than a fixed-ratio baseline?
- **RQ2:** Does target-tokenizer accounting change the reported savings, and does a policy trained for one tokenizer transfer to another?
- **RQ3:** How much does the validator reduce quality failures, and what does it cost in latency and savings? Measured by validator precision and recall against human labels, failure rate with and without the validator, and a validator-frequency sweep.
- **RQ4:** Does hard-span masking reduce constraint violations compared with baselines that do not protect constraints? Measured at prompt level (deterministic) and at output level (IFEval checkers).
- **RQ5 (optional):** Does `prefix_freeze` lower effective cost when provider caching is on?

**Datasets** (200 test items each, plus a separate 20% validation split for tuning)

| Task family | Dataset | Quality metric | Why |
|---|---|---|---|
| Summarization | Arxiv-March23 | ROUGE-L, BERTScore | Same as LLM-DCP |
| Conversation | ShareGPT | ROUGE-L, BERTScore | Same as LLM-DCP |
| Math reasoning | GSM8K (few-shot CoT prompt) | Exact match | Same as LLM-DCP |
| In-context learning | BBH (6 tasks) | Exact match | Same as LLM-DCP |
| RAG / long context | LongBench (HotpotQA, MultiFieldQA subsets) | F1 | Tests the validator and role-aware protection |
| Constraint stress | 150 items from the sets above, each with one IFEval-style verifiable instruction appended | Constraint-violation rate | Tests C1 and C4 |

**Target models (three tokenizers):** one low-cost OpenAI model, plus Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct through Ollama. Temperature 0; all responses cached.

**Baselines:** no compression · random word removal (same ratio as ASPO) · truncation · Selective Context · LLMLingua · LongLLMLingua (RAG set only) · LLMLingua-2 · LLM-DCP · TACO-RL · the pilot loop (M3) without the learned policy.

**Metrics:** TR, CR, $ saved (current list prices), latency (t_ASPO, t_LLM, total, break-even prompt length from formula (6)), task quality, cos(e(P), e(P′)), validator acceptance and fallback rates, validator precision and recall against human labels, hard-span retention (must be 100%), soft-span retention, constraint-violation rate at prompt and output levels, stop-reason distribution, steps taken.

**Ablations:**
- Fixed-ratio stop vs validator stop (RQ1, RQ3)
- Learned policy vs pilot scorer M3
- Hard-span masking on vs off (RQ4)
- Validator on vs off; cosine-only vs cosine + NLI (RQ3)
- Validator frequency f ∈ {1, 2, 4, 8} (cost vs accuracy)
- Batch size k ∈ {1, 4, 16, 64} and the default 5% (latency vs compression)
- Analyzer rules-only vs rules + prototypes
- Optional: with vs without lexical rewrite (C3)

**Statistics:** three seeds for any sampling; 95% bootstrap confidence intervals; paired bootstrap (ROUGE, F1) or McNemar's test (exact match and constraint violations) against the best baseline; cost–quality Pareto curves across the presets and extra ratio settings.

**Integrity rule:** no number goes into the paper unless it is produced by `eval/run_benchmark.py` and saved in `results/`.

---

## 9. Roadmap (relative weeks from now)

| Weeks | Work | Done when |
|---|---|---|
| 1–2 | M0 tokenizer layer; validator with unit tests; analyzer length factor in tokens; two-tier protection mapping; start the 400-prompt labelling. In parallel: fix the paper text (see `PAPER_REVISIONS.md`) | Token counts match official tokenizers; each validator check has a test |
| 3–4 | Removal loop with pilot scorer M3; normalize (M5.1); pilot comparison vs LLMLingua-2 on one dataset | Stop-and-restore works on example prompts; go / no-go decision recorded |
| 5–7 | Policy M4 training with REINFORCE; hard-span masking; reward grid on the validation split; analyzer κ and F1 | Policy beats the pilot scorer on the validation split |
| 8–9 | M7 adapter; M8 proxy, CLI, dashboard | An OpenAI SDK pointed at the proxy works end to end — **usable v1, mid-term demo** |
| 10–13 | Full eval harness; baselines (incl. TACO-RL and LLM-DCP); main runs; ablations; batch and frequency sweeps; statistics | `results/*.csv` and plots generated by script |
| 14–16 | Fill the paper with real results, final edit, submit; viva prep | Paper submitted |

If the go / no-go decision in weeks 3–4 is "no", the paper's claims move to C1, C4 and RQ4, and the compression result is presented as a trade-off.

---

## 10. Out of scope (future work in the paper)

Contextual-bandit or RL budget policy over segments · fine-tuned DeBERTa analyzer · LLM-based rewriting and template restructuring · multilingual "token tax" study · agent-loop mode · user-feedback threshold tuning · full cache-aware optimization beyond `prefix_freeze` · lexical rewriting as a main contribution (kept as an optional extra).

---

## 11. Project structure (target)

```
aspo/
  types.py lexicon.py protected.py segmenter.py     ✅
  analyzer.py                                       ✅ v1
  tokenizers.py                                     ✅
  intake.py validator.py                            ✅
  scorer.py policy.py optimizers.py
  adapters.py pipeline.py __init__.py __main__.py
server.py          # FastAPI OpenAI-compatible proxy
dashboard.py       # Streamlit demo
training/          # policy training (REINFORCE), reward grid search
eval/  datasets.py baselines.py metrics.py run_benchmark.py
data/  analyzer_labels.csv   (400 hand-labelled prompts)
results/           (generated only)
tests/ one test file per module
```

Dependencies stay optional so the core runs without heavy libraries: `[tok]` tiktoken + tokenizers · `[nlp]` sentence-transformers, spaCy, nltk, torch · `[serve]` litellm, fastapi, uvicorn, streamlit · `[eval]` datasets, rouge-score, bert-score, llmlingua.

---

## 12. Progress log

### ✅ M1 Segmenter (2026-11-15)
- `segment(prompt)` accepts a string or OpenAI-style chat messages and returns a `SegmentedPrompt`.
- Assigns 11 roles: system, instruction, context, example, question, format, code, data, history, tool_output, other.
- Detects 12 protected span kinds (code, URL, email, path, placeholder, quoted, date, number, entity, negation, constraint, format). spaCy is optional for entities.
- Reconstruction is exact: `to_text()` / `to_messages()`, with per-segment replacements.
- CLI: `python -m aspo file [--json]`. About 244 ms for 200K characters.

### ✅ M2 Intent & Complexity Analyzer v1 (2026-11-16)
- `analyze(prompt)` returns an `IntentAnalysis`: task (qa, rag_qa, summarization, code, reasoning, extraction, classification, transformation, creative, chitchat), confidence, per-task probabilities, complexity C ∈ [0,1] with its six factors, a feature vector, and human-readable evidence signals.
- Method: lexical cues weighted by the role of the segment they occur in, plus structural features, combined in a linear model with softmax. QA is the residual class.
- CLI: `python -m aspo.analyzer file [--json]`.
- Pending: prototype-embedding scorer, κ/F1 evaluation. (Token-based length done with M0.)

### ✎ Plan revision (2026-11-18)
- Architecture changed to the validator-controlled learned removal policy (§0, §4, §5, §6, §7). No code changed in this revision.

### ✎ Plan decisions (2026-11-20)
- Formula references aligned with §6. Removal unit = words; training = one-shot REINFORCE; reward gate = H ∧ Q ∧ cos ≥ τ(C); start from LLMLingua-2 mBERT-base; laptop for development, Kaggle for full training. Soft spans are atomic removal units; in instruction-type segments only the detected spans are hard, other words are removable under NLI. No code changed.

### ✅ M0 Tokenizer layer (2026-11-25)
- `count_tokens(text, model)` gives T_M. tiktoken for OpenAI models (exact; unknown GPT-like names assume o200k_base and are flagged). Hugging Face `tokenizers` for Llama-3.x and Qwen2/2.5 (exact; Llama is gated and needs `HF_TOKEN`, otherwise it falls back to the approximation and is flagged). Claude, Gemini and unknown models: o200k_base count × a per-family factor (1.0 until `calibrate()` is fitted against the provider's counting endpoint), `exact=False`.
- `tokenizer_info(model)` says which tokenizer is used and whether it is exact. `encode()` returns ids (exact backends only). `register_tokenizer()` plugs in custom backends (for example a provider endpoint).
- `price_per_token(model)` reads LiteLLM's **bundled** cost table (reproducible, no network), with input, output and cache-read prices. `ollama/*` is free. `prompt_cost()` and `savings()` report TR, CR and $ saved, with the tokenizer and price source.
- LRU cache on (model, text) counts (16,384 entries): 0.8 ms for 5.8K tokens uncached, about 1 µs cached.
- Counts are raw text with no BOS and no chat-template overhead (constant per message, so it cancels in before/after).
- Analyzer: `AnalyzerConfig(target_model=...)` measures the length factor in T_M (saturates at 5,000 tokens), adds `total_tokens` to features, and the CLI takes `-m`.
- CLI: `python -m aspo.tokenizers file -m gpt-4o -m ollama/qwen2.5 ...`.
- Extras: `[tok]` = tiktoken + tokenizers; `[serve]` = litellm.

### ✅ M1 additions: two-tier protection + intake (2026-11-26)
- `Tier` (HARD/SOFT) on every `ProtectedSpan`; `span_tier(kind, role)` implements §5 M1 exactly. Decision: spans in OTHER segments are hard (an unclassified paragraph may be the task). A soft span overlapping a hard span is promoted to hard.
- `Segment.removal_units()`: words, minus every word touching a hard span; the words of a soft span (or of overlapping soft spans) form one atomic unit. This is the unit M3/M4/M5 will score and remove.
- `aspo/intake.py`: `check_intake(prompt, model, max_tokens=None)`. Empty prompts raise `EmptyPromptError`. Decision: the limit is the model's context window (`context_window(model)` in M0, LiteLLM `max_input_tokens`); a longer prompt returns `optimise=False, reason="too_long"` and is sent unchanged.

### ✅ M6 Validator (2026-11-27)
- `Validator.validate(sp, replacements, C)` implements formula (9): H → Q → S → N, short-circuiting at the first failure (`short_circuit=False` computes every value). `gate()` = H ∧ Q ∧ S (reward gate G, formula (12)). `similarity()` exposes cos(e(P), e(P′)). `final_check(sp, replacements, output)` re-runs H and Q on the rebuilt text or messages.
- S uses the chunked mean e(P) = Σ n_c·e(c) / Σ n_c over chunks below 254 word pieces per segment; chunk embeddings are cached, so unchanged segments are free. τ(C) = 0.80 + 0.15·C (C clamped).
- N: bidirectional entailment with `cross-encoder/nli-deberta-v3-small` (labels: contradiction / entailment / neutral, softmax over logits) on changed INSTRUCTION, QUESTION, FORMAT, SYSTEM and OTHER segments. Results cached.
- `ValidationResult`: passed, first failed check (stop reason), and for each check its value, threshold, details and time; soft-span retention as a metric.
- Real models are loaded lazily (`import aspo` stays at 0.07 s). PyTorch 2.11 + CUDA 12.8 and sentence-transformers 6.3 are in `.venv`; the RTX 4050 is used.
- Measured (RTX 4050, warm): H and Q < 0.1 ms; S ≈ 40–50 ms per step; N ≈ 100 ms per changed instruction segment (pairs batched). Cold start: MiniLM ≈ 2 s, DeBERTa ≈ 10 s.
- **Finding (risk confirmed):** the backward NLI direction (P′ ⊨ P) is unreliable on imperatives. "Summarize the following article in 3 bullet points." → "Summarize the article in 3 bullet points." scored 0.011, while the more telegraphic "Summarize article 3 bullet points." scored 0.455. Dropping politeness ("Could you please kindly …") passed at 0.96–0.99. With threshold 0.8, N will reject most instruction edits. 

### ✅ M6 NLI decision: raw text, threshold 0.05 (2026-11-27)
- Probe `eval/nli_probe.py`: 48 hand-labelled instruction edits made by deletion only. 24 keep the task (politeness, filler, articles, "following"); 24 lose part of it (a dropped qualifier, object, audience or language: "short", "five", "Python", "formal", "in simple terms" …). Scored with DeBERTa-v3-small: raw text vs three statement rewrites applied to both sides ("The task is to …", "The user asks the assistant to …", 'The user said: "…"').
- Separation (AUC of min(fwd, bwd), ok vs bad): raw 0.990, task 0.979, asks 0.984, said 0.990. The NLI score already ranks the edits almost perfectly; the 0.8 threshold was the problem, not the model.
- Keep / leak (ok edits passed / bad edits passed) at thresholds 0.01 · 0.02 · **0.05** · 0.1 · 0.5 · 0.8: raw 92/8 · 88/4 · **88/0** · 83/0 · 71/0 · 46/0; the best rewrite (said) 83/4 · 75/0 · 75/0 · 71/0 · 50/0 · 50/0. Rewriting is worse at every threshold: it lowered several good edits (e.g. "translate the sentence into French" from 0.90 to 0.004 under "task").
- **Decision: option 2 (lower the threshold) on raw text; option 4 (statement rewrite) rejected.** Default `nli_threshold = 0.05`: 88% of good edits pass, 0% of bad. Bad edits score 0.000–0.037; the 3 good edits still rejected ("Summarize the article in 3 bullet points." 0.011, "Summarize article in 3 bullet points." 0.008, "include a brief explanation…" 0.006) are a conservative miss, not a leak.
- The forward direction (P ⊨ P′) is ≈1 for every deletion (AUC 0.65), so on deletions N is decided by the backward direction. Forward is kept because it costs nothing (batched) and guards the optional rewrite (M5.3), which can add words.
- Caveat: 48 pairs, written by us, threshold chosen on the same set. Re-tune on the validation split with real loop candidates (BACKLOG M6).

**Tests:** 328 passing (48 segmenter/protected-span + 33 analyzer + 43 tokenizer + 151 tiers + 18 intake + 35 validator, 4 of which load the real models and are marked `slow`).
