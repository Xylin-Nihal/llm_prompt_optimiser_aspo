# ASPO: Adaptive Semantic Prompt Optimization Framework

Model-agnostic middleware that cuts the number of tokens in an LLM prompt while keeping its meaning. How hard it compresses adapts to the task, to how much each part of the prompt matters, and to the target model's tokenizer.

## Pipeline

| # | Module | Status |
|---|--------|--------|
| 1 | Segmenter and protected-span detection (`aspo/segmenter.py`, `aspo/protected.py`) | ✅ done |
| 2 | Intent and task analyzer | ⏳ |
| 3 | Semantic importance scorer | ⏳ |
| 4 | Strategy selector / budget controller | ⏳ |
| 5 | Optimizers: restructure, dedup, tokenizer-aware lexical, compress | ⏳ |
| 6 | Quality validator with fallback | ⏳ |
| 7 | Model adapter, proxy and dashboard | ⏳ |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"
# optional, gives better named-entity protection:
pip install -e ".[nlp]" && python -m spacy download en_core_web_sm
pytest
```

## Module 1: Segmenter

The segmenter splits a prompt string, or a list of OpenAI-style chat `messages`, into segments. Each segment gets a **role** and a list of **protected spans**.

```python
from aspo import segment

sp = segment(open("examples/rag_prompt.txt").read())
for s in sp.segments:
    print(s.role.value, s.header, [p.kind.value for p in s.protected])

# Rebuilding is exact, and later modules swap in a new text per segment:
assert sp.to_text() == original
compressed = sp.to_text({1: "Apple Q4-24 revenue $94.9 billion (+6%)."})   # "" drops a segment
```

To see the segmentation from the command line:

```bash
python -m aspo examples/rag_prompt.txt          # table view
python -m aspo examples/rag_prompt.txt --json   # full JSON
python -m aspo examples/chat.json               # chat messages
```

**Roles:** `system, instruction, context, example, question, format, code, data, history, tool_output, other`.

**How roles are decided**, strongest signal first:
1. Explicit structure: code fences, XML-style tag blocks (`<document>…</document>`), markdown headers, `Label:` lines. A section role carries over to later paragraphs until the next header.
2. Few-shot pairs (`Q:/A:`, `Input:/Output:`, `Review:/Sentiment:`). A pair with an empty output is the open query.
3. Content rules: imperative verbs, question marks, persona phrases ("You are…"), JSON/CSV/code shape, length.
4. Position: a trailing request at the end of a long final paragraph gets its own segment.

In chat input, system messages map to `system`, earlier turns to `history`, tool messages to `tool_output`, and the last user message is segmented in full.

**Protected span kinds:** `code, url, email, path, placeholder, quoted, date, number, entity, negation, constraint, format`. Later stages must never delete these spans.
- **Negations** protect the whole clause around them.
- **Constraint and format** sentences ("must be under 300 words", "respond in JSON") are protected only in instruction-like roles. Inside documents, words like "should" and "table" are ordinary content.
- **Code blocks and answer cues** (a trailing `A:`) are protected in full.

All span offsets are relative to `segment.text`.
