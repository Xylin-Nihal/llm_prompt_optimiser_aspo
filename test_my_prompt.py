#!/usr/bin/env python3
"""
Interactive prompt testing tool for ASPO segmenter.

Usage:
    python test_my_prompt.py                    # Interactive mode
    python test_my_prompt.py path/to/prompt.txt # Test a file
    python test_my_prompt.py --examples         # Show example prompts
"""

import sys
import json
import io
from pathlib import Path
from aspo import segment, SegmentRole as R, SpanKind

# Fix UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', newline='')

def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")

def analyze_prompt(text, name="Your Prompt"):
    """Analyze and display segmentation results."""
    print_section(f"Analyzing: {name}", "=")

    sp = segment(text)

    # Basic stats
    print(f"\n[STATISTICS]")
    print(f"   Total characters: {len(text):,}")
    print(f"   Total words: {len(text.split()):,}")
    print(f"   Number of segments: {len(sp.segments)}")
    print(f"   Total protected spans: {sum(len(s.protected) for s in sp.segments)}")
    print(f"   Is chat: {sp.is_chat}")

    # Role distribution
    role_counts = sp.role_counts()
    if role_counts:
        print(f"\n[ROLE DISTRIBUTION]")
        for role, count in sorted(role_counts.items()):
            print(f"   {role:15} : {count:2} segment(s)")

    # Detailed segments
    print(f"\n[SEGMENTS DETAIL]")
    print("-" * 70)

    for seg in sp.segments:
        # Header
        msg_info = f" (message {seg.message_index})" if seg.message_index is not None else ""
        print(f"\n[{seg.id}] {seg.role.value.upper()}{msg_info}")

        # Content preview
        preview = seg.text.replace("\n", " ⏎ ")
        if len(preview) > 70:
            preview = preview[:67] + "..."
        print(f"    Text: {preview}")

        # Stats
        pct = seg.protected_ratio * 100
        print(f"    Stats: {seg.word_count} words, {pct:.0f}% protected")

        # Header if any
        if seg.header:
            print(f"    Header: {seg.header}")

        # Protected spans
        if seg.protected:
            print(f"    Protected spans ({len(seg.protected)}):")
            for ps in seg.protected:
                kind_name = ps.kind.value
                # Truncate long text
                txt = ps.text
                if len(txt) > 50:
                    txt = txt[:47] + "..."
                print(f"      • {kind_name:12} | {txt!r}")
        else:
            print(f"    Protected spans: none")

    # Verification
    print_section("Verification", "-")
    reconstructed = sp.to_text()
    if reconstructed == text:
        print("✅ Exact text reconstruction: PASS")
    else:
        print("❌ Exact text reconstruction: FAIL")
        print(f"   Original length: {len(text)}")
        print(f"   Rebuilt length:  {len(reconstructed)}")

    # Check all spans are valid
    all_valid = True
    for seg in sp.segments:
        for ps in seg.protected:
            if seg.text[ps.start:ps.end] != ps.text:
                print(f"❌ Invalid span offset in segment {seg.id}")
                all_valid = False

    if all_valid:
        print("✅ All span offsets valid: PASS")
    else:
        print("❌ Some span offsets invalid: FAIL")

def show_examples():
    """Show built-in test prompts."""
    examples = {
        "RAG with Markdown": """You are a helpful financial analyst.

### Context
Apple Inc. reported revenue of $94.9 billion for Q4 2024. The company is very profitable.

### Question
How much revenue did Apple report?

Answer in 2 bullet points. Do not speculate about future results.""",

        "Inline Labels": """Context: The Amazon rainforest is about 5.5 million square kilometers.
Question: How large is the Amazon?
Answer:""",

        "XML Structure": """Analyze this document:

<document>
The company was founded in 2007. It now has 50,000 employees.
Headquarters: Mountain View, California.
</document>

<format>
JSON format with keys: year, employees, location
</format>""",

        "Few-Shot Examples": """Classify the emotion:

Text: "I'm so excited about this!"
Emotion: joy

Text: "This is terrible."
Emotion: sadness

Text: "That's awesome!"
Emotion:""",

        "Code Analysis": """Why does this Python code fail?

```python
def divide(a, b):
    return a / b

result = divide(10, 0)
```

Explain what happens and suggest a fix.""",

        "Casual Chat": """Hey! I'm trying to build a machine learning model to predict customer churn. I've collected data from usage logs, support tickets, and billing history. So far I have about 100K samples but I'm not sure if that's enough. Could you help me figure out if this dataset is sufficient or if I need more data? Also, what are the key features I should focus on?""",

        "Long Context": """The United Nations is a global organization founded in 1945 after World War II. It has 193 member states and aims to promote peace and international cooperation. The UN has six main organs: the General Assembly, Security Council, Economic and Social Council, Trusteeship Council, International Court of Justice, and the Secretariat. The Secretary-General is the chief administrative officer. The organization works on issues including climate change, poverty, humanitarian aid, and conflict resolution. What are the main achievements of the UN?""",
    }

    for name, text in examples.items():
        analyze_prompt(text, name)
        input("\n[Press Enter to see next example...]")

def interactive_mode():
    """Interactive prompt testing."""
    print_section("ASPO Segmenter - Interactive Testing")
    print("""
Welcome! You can:
  1. Enter a prompt directly
  2. Load from a file
  3. See examples
  4. Exit

Type 'help' for detailed instructions.
    """)

    while True:
        print("\n" + "=" * 70)
        choice = input("Choose option (1-4, or 'help'): ").strip().lower()

        if choice in ('help', '?'):
            print("""
Options:
  1 - Type or paste your prompt directly
  2 - Load from a .txt or .json file
  3 - See 7 example prompts
  4 - Exit

Examples:
  python test_my_prompt.py path/to/prompt.txt
  python test_my_prompt.py --examples
            """)

        elif choice == '1':
            print("\nEnter your prompt (press Ctrl+D on Unix or Ctrl+Z on Windows when done):")
            print("(Or type 'END' on a new line)")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                except EOFError:
                    break

            if lines:
                text = "\n".join(lines)
                analyze_prompt(text, "User Input")

        elif choice == '2':
            path = input("Enter file path: ").strip()
            try:
                p = Path(path)
                if p.suffix == '.json':
                    with open(p) as f:
                        messages = json.load(f)
                    sp = segment(messages)
                    print(f"✅ Loaded {len(messages)} chat messages")
                else:
                    text = p.read_text()
                    analyze_prompt(text, f"File: {path}")
            except FileNotFoundError:
                print(f"❌ File not found: {path}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == '3':
            show_examples()

        elif choice == '4':
            print("\nGoodbye! 👋")
            break

        else:
            print("Invalid choice. Try again.")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg in ('--examples', '-e'):
            show_examples()
        elif arg in ('--help', '-h'):
            print(__doc__)
        else:
            # Treat as file path
            try:
                path = Path(arg)
                if path.suffix == '.json':
                    with open(path) as f:
                        messages = json.load(f)
                    sp = segment(messages)
                    print(f"Chat messages ({len(sp.segments)} segments):")
                    for s in sp.segments:
                        print(f"  [{s.message_index}] {s.role.value:12} | {s.text[:60]}")
                else:
                    text = path.read_text()
                    analyze_prompt(text, f"File: {arg}")
            except FileNotFoundError:
                print(f"❌ File not found: {arg}")
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
