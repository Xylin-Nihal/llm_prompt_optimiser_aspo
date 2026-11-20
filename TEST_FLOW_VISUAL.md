# ASPO Testing Flow - Visual Guide

## 🎯 Complete Testing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     YOUR TEST PROMPT                                 │
│  (plain text, chat JSON, file, or interactive input)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼─────┐
                    │  SEGMENTER │
                    └──────┬─────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
  ┌──────────────┐                   ┌──────────────┐
  │   SEGMENTS   │                   │ PROTECTED    │
  │   (roles)    │                   │   SPANS      │
  └──────┬───────┘                   └──────┬───────┘
         │                                   │
         │  Examples:                        │  Examples:
         │  • SYSTEM (4 words)               │  • CODE (100%)
         │  • CONTEXT (42 words)             │  • NUMBER (5)
         │  • QUESTION (12 words)            │  • NEGATION (1)
         │  • FORMAT (8 words)               │  • CONSTRAINT (2)
         │                                   │
         └──────────────┬────────────────────┘
                        │
         ┌──────────────▼─────────────────┐
         │   VERIFICATION TESTS           │
         └──────────────┬─────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    ┌────────┐    ┌──────────┐    ┌────────┐
    │EXACT   │    │PROTECTED │    │COVERAGE│
    │REBUILD │    │OFFSETS   │    │        │
    │        │    │VALID     │    │NO GAP  │
    └────┬───┘    └────┬─────┘    └────┬───┘
         │             │              │
         └─────────────┼──────────────┘
                       │
                   ✅ or ❌
                       │
        ┌──────────────▼──────────────┐
        │      RESULTS DISPLAY        │
        │                             │
        │  • Role summary            │
        │  • Protected span list      │
        │  • Problem highlighting    │
        │  • Compression estimate    │
        └─────────────────────────────┘
```

---

## 📊 Testing Process - Step by Step

### Phase 1: Input (3 options)

#### Option A: Interactive
```
$ python test_my_prompt.py
  ↓
  Choose option 1
  ↓
  Paste or type your prompt
  ↓
  Analysis
```

#### Option B: Command Line
```
$ python -m aspo my_prompt.txt
  ↓
  Instant table view
```

#### Option C: Python
```python
from aspo import segment
sp = segment("your prompt")
```

---

### Phase 2: Segmentation Process

Your input gets split by detecting:

```
Raw Prompt
│
├─ Explicit Structure?
│  ├─ Markdown headers (### Section)
│  ├─ XML tags (<tag>...</tag>)
│  ├─ Inline labels (Label: content)
│  └─ Code fences (```...```)
│
├─ Few-Shot Pattern?
│  ├─ Q:/A: pairs
│  ├─ Input:/Output: pairs
│  └─ Empty output = open query
│
└─ Content Heuristics?
   ├─ Imperative verb? → INSTRUCTION
   ├─ Question mark? → QUESTION
   ├─ Long text? → CONTEXT
   ├─ Looks like code? → CODE
   └─ Looks like JSON? → DATA
```

**Output:** List of segments with roles

---

### Phase 3: Protected Span Detection

For each segment, find critical content:

```
Segment Text
│
├─ CODE
│  └─ ```...``` or `code()`
│
├─ URLS/EMAILS/PATHS
│  └─ https://..., user@..., /path/...
│
├─ NUMBERS
│  └─ 123, $94.9B, v1.2.3, 300ms
│
├─ DATES
│  └─ 2024-01-01, January 15, 10:30 AM
│
├─ ENTITIES
│  └─ Apple Inc, Paris, NASA (capitalized)
│
├─ NEGATIONS
│  └─ "Do not...", "Never...", "Must not..."
│
├─ CONSTRAINTS
│  └─ "Must be...", "Only use...", "Max 300 words"
│
└─ FORMAT
   └─ "Respond in JSON", "2 bullet points"
```

**Output:** Protected spans with offsets relative to segment

---

### Phase 4: Verification

Three critical checks:

```
Check 1: Text Reconstruction
┌─────────────────────────────────┐
│ original_text == rebuilt_text?  │
│         YES → ✅ PASS            │
│          NO → ❌ FAIL (bug!)     │
└─────────────────────────────────┘

Check 2: Span Offsets Valid
┌──────────────────────────────────────────┐
│ for each span:                           │
│   segment.text[span.start:span.end]      │
│   == span.text?                          │
│         YES → ✅ PASS                     │
│          NO → ❌ FAIL (wrong offsets)    │
└──────────────────────────────────────────┘

Check 3: Full Coverage
┌──────────────────────────────────────────┐
│ All text accounted for in segments?      │
│ No gaps? No overlaps?                    │
│         YES → ✅ PASS                     │
│          NO → ❌ FAIL (broken prompt)    │
└──────────────────────────────────────────┘
```

---

### Phase 5: Display Results

```
======================================================================
  Analyzing: Your Prompt
======================================================================

📊 Statistics:
   Total characters: 456
   Total words: 87
   Number of segments: 4
   Total protected spans: 12
   Is chat: False

📋 Role Distribution:
   system        :  1 segment(s)
   context       :  2 segment(s)
   question      :  1 segment(s)

📝 Segments Detail:
----------------------------------------------------------------------

[0] SYSTEM
    Text: You are a helpful financial analyst assistant...
    Stats: 7 words, 0% protected
    Protected spans: none

[1] CONTEXT
    Text: ### Context
         Apple Inc. reported $94.9 billion...
    Stats: 42 words, 43% protected
    Protected spans (3):
      • entity        | 'Apple Inc'
      • number        | '$94.9 billion'
      • negation      | 'Never speculate on future results'

[2] QUESTION
    Text: How much revenue did Apple report?
    Stats: 7 words, 15% protected
    Protected spans (1):
      • entity        | 'Apple'

[3] FORMAT
    Text: Answer in exactly 2 bullet points...
    Stats: 8 words, 87% protected
    Protected spans (3):
      • constraint    | 'exactly 2'
      • format        | 'bullet points'
      • number        | '2 bullet'

======================================================================
Verification
----------------------------------------------------------------------
✅ Exact text reconstruction: PASS
✅ All span offsets valid: PASS
```

---

## 🔄 Real Example: Full Testing Flow

### Input Prompt
```
You are a Python expert.

### Context
Here's a function that crashes. It has 500 lines and took 2 weeks to write.

```python
def divide(a, b):
    return a / b
```

### Question
Why does this fail? Never suggest rewriting.

Respond in JSON format only.
```

### Segmentation Process

| Step | Detection | Result |
|------|-----------|--------|
| 1 | First line: "You are" → SYSTEM | Segment 0: SYSTEM |
| 2 | "### Context" → Header | Segment 1: CONTEXT (header) |
| 3 | Paragraph after header → Inherits CONTEXT | Segment 1: CONTEXT (expanded) |
| 4 | ` ``` ` fence detected | Segment 2: CODE (fully protected) |
| 5 | "### Question" → Header | Segment 3: QUESTION (header) |
| 6 | "Never suggest" → Contains NEGATION | Segment 3: QUESTION (protected) |
| 7 | "Respond in JSON" → FORMAT keywords | Segment 4: FORMAT |

### Protected Spans Detected

| Segment | Spans Found |
|---------|-------------|
| SYSTEM | none |
| CONTEXT | entity: "Python"; number: "500"; number: "2 weeks" |
| CODE | code: (entire function) |
| QUESTION | negation: "Never suggest rewriting" |
| FORMAT | format: "JSON format only" |

### Output Verification
```
✅ Reconstruction: original == rebuilt
✅ Offsets: all spans valid
✅ Coverage: 100% of text accounted for
```

---

## 🧪 Unit Test Coverage

### How Tests Work

```
Test Suite (96 tests)
│
├─ Protected Span Tests (48)
│  ├─ Offset accuracy
│  ├─ Code detection
│  ├─ URL/Email/Path detection
│  ├─ Number detection (with units)
│  ├─ Date detection
│  ├─ Placeholder detection
│  ├─ Quoted string detection
│  ├─ Negation detection
│  ├─ Constraint detection
│  ├─ Format detection
│  ├─ Entity detection
│  ├─ Filtering by kind
│  └─ No duplicates
│
└─ Segmentation Tests (48)
   ├─ Exact reconstruction (parametrized)
   ├─ Span offsets relative to segment (parametrized)
   ├─ Markdown structure
   ├─ Inline labels
   ├─ XML tags
   ├─ Few-shot pairs
   ├─ Code blocks
   ├─ Trailing questions
   ├─ JSON data
   ├─ Constraint protection role-awareness
   ├─ Chat format
   ├─ Message roundtrip
   └─ Empty/whitespace handling
```

### Running Tests

```
Test Execution Flow
│
├─ pytest discovers tests/
│  ├─ test_protected.py
│  └─ test_segmenter.py
│
├─ Each test file imports aspo
│  ├─ Loads types.py
│  ├─ Loads lexicon.py
│  ├─ Loads protected.py
│  └─ Loads segmenter.py
│
├─ Tests execute in order
│  ├─ Import checks
│  ├─ Data validation
│  ├─ Accuracy checks
│  └─ Edge case tests
│
└─ Results
   ├─ 96 PASSED ✅
   ├─ 0 FAILED ❌
   ├─ 0 SKIPPED ⏭️
   └─ Runtime: 0.11s
```

### Test Output Example

```bash
$ .venv\Scripts\python -m pytest tests/ -v

test_protected.py::test_offsets_match_text PASSED                     [  2%]
test_protected.py::test_code_fences_and_inline_code PASSED            [  4%]
test_protected.py::test_numbers_inside_code_are_not_duplicated PASSED [  6%]
test_protected.py::test_urls_emails_paths PASSED                      [  8%]
...
test_segmenter.py::test_reconstruction_is_exact[rag_markdown] PASSED   [ 48%]
test_segmenter.py::test_reconstruction_is_exact[labels_inline] PASSED  [ 50%]
...
test_segmenter.py::test_to_text_on_chat_raises PASSED                 [ 100%]

============================== 96 passed in 0.11s ==============================
```

---

## 📈 Quality Metrics

```
Coverage by Category
│
├─ Span Types: 12/12 (100%)
│  └─ Code, URL, Email, Path, Placeholder, Quoted, Date,
│     Number, Entity, Negation, Constraint, Format
│
├─ Prompt Formats: 8/8 (100%)
│  └─ RAG markdown, inline labels, XML tags, few-shot,
│     code, plain long, JSON, chat messages
│
├─ Test Types: 4/4 (100%)
│  └─ Property-based, example-based, edge-case, parametrized
│
├─ Infrastructure: 3/3 (100%)
│  └─ Exact reconstruction, offset validation, coverage check
│
└─ Total: 96 tests × 100% = ✅ All Pass
```

---

## 🎓 Testing Workflow for Your Research Paper

### Week 1-2: Understand Testing
```
Day 1-2: Run test_my_prompt.py --examples
Day 3-4: Test 10 of your own prompts
Day 5: Review protected spans for each
Days 6-7: Understand why each span is protected
```

### Week 3-4: Extend Tests
```
Day 1-3: Add tests for your specific prompt types
Day 4-5: Test edge cases in your domain
Days 6-7: Validate role classification accuracy
```

### Week 5+: Research Analysis
```
Document 1: What % of prompts segment correctly?
Document 2: What % of protected spans are accurate?
Document 3: False positive/negative rates per span type
Document 4: Variance in segmentation across prompt formats
```

### Paper Results
```
Table 1: Segmentation accuracy on 100 test prompts
Table 2: Protected span recall and precision
Figure 1: Role distribution across dataset
Figure 2: Protected content by prompt type
```

---

## ✨ Summary

**Testing covers:**
- 12 protected span types
- 8 prompt formats
- 96 unit tests
- Exact reconstruction guarantee
- Offset validation
- Edge cases

**You can test:**
- Interactive menu
- Command line
- Python API
- File input (txt, json)

**All tests pass:** ✅ 96/96 (0.11s)

**Your prompts will:**
- Segment accurately
- Protect critical content
- Reconstruct exactly
- Enable safe compression

---

Generated: 2026-11-16
