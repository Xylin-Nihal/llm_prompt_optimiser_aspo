"""Small hand-curated word lists used by the rule-based parts of ASPO."""

from __future__ import annotations

IMPERATIVE_VERBS = frozenset(
    """
    act add adjust analyse analyze answer apply argue arrange ask assess assume avoid
    break brainstorm build calculate categorise categorize change check choose cite
    clarify classify combine compare compile complete compose compute condense
    consider construct continue convert correct count create critique debug decide
    define delete describe design detect determine develop diagnose discuss do draft
    draw edit elaborate enumerate estimate evaluate examine explain explore express
    extract figure fill find finish fix focus follow format generate get give go group
    guess help highlight identify ignore illustrate imagine implement improve include
    infer insert interpret investigate keep label let list locate make map mark match
    measure mention merge modify name note optimise optimize order organise organize
    outline paraphrase pick plan point predict prepare present prioritise prioritize
    produce proofread propose provide put rank rate read recommend refactor reformat
    remember remove rename reorder rephrase replace reply report represent research
    respond restate restructure return review revise rewrite run score select separate
    shorten show simplify solve sort specify split state suggest summarise summarize
    tell test think transform translate treat try turn use validate verify walk write
    """.split()
)

#: Verbs that introduce an output-format instruction rather than a task.
FORMAT_VERBS = frozenset({"respond", "reply", "return", "format", "output", "answer", "structure"})

QUESTION_WORDS = frozenset(
    """
    what why how who whom whose which where when whether is are was were do does did
    can could would will should shall may might has have had am
    """.split()
)

#: Capitalised words that should not be treated as named entities on their own.
ENTITY_STOPWORDS = frozenset(
    """
    I A An The This That These Those It Its You Your Yours We Our They Their He She His
    Her Him Me My Mine Us Them Please If When What Why How Who Which Where Whether Do
    Does Did Is Are Was Were Be Been Can Could Would Will Should Shall May Might Must
    And But Or Nor For So Yet In On At To Of As By From With Without About Into Over
    Under After Before Then Also Note Yes No Not Ok Okay Here There Now Today Each Every
    All Any Some Many Most Such Both Either Neither One Two Three First Second Third
    Next Finally Example Examples Input Output Answer Question Context Task Instructions
    Hi Hello Hey Thanks Thank Dear Sure Let Make Use Keep Given Based Assume Consider
    """.split()
)

PERSONA_PREFIXES = (
    "you are",
    "you're",
    "act as",
    "acting as",
    "pretend you are",
    "pretend to be",
    "imagine you are",
    "assume the role",
    "as an ai",
    "as a ",
    "as an ",
    "your role is",
    "you will act",
    "you will be acting",
)

REQUEST_PREFIXES = (
    "please",
    "kindly",
    "i want you to",
    "i need you to",
    "i would like you to",
    "i'd like you to",
    "i need",
    "i want",
    "your task is",
    "your job is",
    "the task is",
    "your goal is",
    "help me",
    "let's",
)

POLITE_QUESTION_REQUEST = (
    "can you",
    "could you",
    "would you",
    "will you",
    "can u",
    "would you mind",
    "is it possible to",
    "could you please",
    "can you please",
)

#: Section-label keyword -> role name (role names match ``SegmentRole`` values).
SECTION_KEYWORDS: dict[str, str] = {
    # instruction
    "instruction": "instruction", "instructions": "instruction", "task": "instruction",
    "tasks": "instruction", "objective": "instruction", "goal": "instruction",
    "directions": "instruction", "guidelines": "instruction", "rules": "instruction",
    "requirements": "instruction", "constraints": "instruction", "steps": "instruction",
    "prompt": "instruction", "request": "instruction",
    # system / persona
    "system": "system", "persona": "system", "role": "system",
    # context
    "context": "context", "background": "context", "document": "context",
    "documents": "context", "passage": "context", "passages": "context",
    "article": "context", "text": "context", "reference": "context",
    "references": "context", "source": "context", "sources": "context",
    "knowledge": "context", "information": "context", "info": "context",
    "transcript": "context", "notes": "context", "email": "context",
    "content": "context", "story": "context", "abstract": "context",
    "excerpt": "context", "report": "context", "paper": "context", "essay": "context",
    "description": "context", "scenario": "context", "situation": "context",
    # examples
    "example": "example", "examples": "example", "sample": "example",
    "samples": "example", "demonstration": "example", "demonstrations": "example",
    "illustration": "example", "shot": "example",
    # question
    "question": "question", "questions": "question", "query": "question",
    "q": "question", "problem": "question", "ask": "question",
    # format
    "format": "format", "output": "format", "response": "format",
    "schema": "format", "answer": "format", "a": "format", "expected": "format",
    "style": "format", "tone": "format",
    # code
    "code": "code", "snippet": "code", "script": "code", "function": "code",
    "program": "code", "error": "code", "traceback": "code", "stacktrace": "code",
    # data
    "data": "data", "json": "data", "csv": "data", "table": "data",
    "dataset": "data", "input": "data", "inputs": "data", "records": "data",
}

#: Labels opening the *input* half of a few-shot pair.
PAIR_INPUT_LABELS = frozenset(
    "q question input user human text review sentence customer prompt query problem".split()
)
#: Labels opening the *output* half of a few-shot pair.
PAIR_OUTPUT_LABELS = frozenset(
    "a answer output assistant ai bot label sentiment category agent response solution".split()
)

SMALL_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
    "thirty|fifty|hundred|single|a\\s+few|a\\s+couple\\s+of"
)

#: Intent-analyzer cues: task name (``TaskType`` value) -> [(regex, weight)].
#: Patterns are matched case-insensitively against each segment's body; the
#: weight is scaled by how task-bearing that segment's role is (see analyzer).
TASK_CUES: dict[str, list[tuple[str, float]]] = {
    "summarization": [
        (r"\bsummar(?:y|ies|ise|ize|ised|ized|ising|izing)\b", 3.0),
        (r"\btl;?dr\b", 3.0),
        (r"\b(?:key|main) (?:points|takeaways|ideas|findings)\b", 1.5),
        (r"\b(?:condense|recap|gist|synopsis|overview of|shorten)\b", 1.5),
    ],
    "code": [
        (r"\b(?:code|function|method|class|script|program|bug|debug|refactor|compile|"
         r"traceback|stack ?trace|exception|unit tests?|api|endpoint|regex|algorithm)\b", 1.5),
        (r"\b(?:python|javascript|typescript|java|golang|rust|sql|bash|html|css|react|"
         r"django|flask|numpy|pandas|kotlin|swift|php|ruby)\b|(?<!\w)(?:c\+\+|c#)(?!\w)", 1.5),
        (r"\bimplement(?:ation|ing|s)?\b", 1.0),
    ],
    "reasoning": [
        (r"\b(?:solve|calculate|compute|prove|derive|equation|probability|integral|"
         r"derivative|arithmetic|algebra|geometry)\b", 2.0),
        (r"\b(?:math|maths|mathematics|word problem)\b", 1.5),
        (r"\bstep[- ]by[- ]step\b", 1.0),
        (r"\bhow many\b", 1.0),
        (r"\b(?:puzzle|riddle|logical(?:ly)?|deduce)\b", 1.0),
    ],
    "extraction": [
        (r"\bextract(?:s|ed|ing|ion)?\b", 3.0),
        (r"\b(?:pull out|list (?:all|every|each)|find (?:all|every)|identify (?:all|every|each)|parse)\b", 1.5),
        (r"\b(?:named entities|entities|fields|key[- ]value|attributes)\b", 1.0),
    ],
    "classification": [
        (r"\b(?:classify|classification|categori[sz]e|categori[sz]ation)\b", 3.0),
        (r"\b(?:sentiment|spam|label|category|categories)\b", 1.5),
        (r"\bpositive,? (?:or )?negative\b|\b(?:true|yes) or (?:false|no)\b", 1.0),
        (r"\bwhich (?:category|class|label)\b", 1.0),
    ],
    "transformation": [
        (r"\btranslat(?:e|es|ed|ing|ion)\b", 3.0),
        (r"\b(?:rewrite|rephrase|paraphrase|proofread|reword|convert|transform)\b", 2.5),
        (r"\b(?:fix|correct) (?:the |my )?(?:grammar|spelling|typos|punctuation)\b", 2.5),
        (r"\b(?:make (?:it|this) (?:more|less|sound)|in an? (?:formal|informal|casual|professional|friendly) tone)\b", 1.5),
        (r"\binto (?:english|french|spanish|german|hindi|chinese|japanese|arabic|tamil|telugu|bengali|italian|portuguese)\b", 1.0),
    ],
    "creative": [
        (r"\b(?:story|stories|poem|poetry|haiku|limerick|lyrics|song|novel|fiction|screenplay|joke|slogan|tagline)\b", 2.5),
        (r"\b(?:creative|imaginative|fictional)\b", 1.5),
        (r"\b(?:brainstorm|come up with|invent)\b", 1.0),
        (r"\b(?:blog post|essay|tweet|caption|speech|cover letter)\b", 1.0),
        (r"\bwrite (?:me )?an?\b", 0.75),
        (r"\b(?:plan|itinerary|outline|ideas|suggestions|recommendations)\b", 1.0),
    ],
    "chitchat": [
        (r"^\W*(?:hi|hello|hey|hiya|yo|good (?:morning|afternoon|evening)|greetings)\b", 2.0),
        (r"\bhow are you\b|\bwhat'?s up\b|\bhow'?s it going\b", 2.0),
        (r"^\W*(?:thanks|thank you|thx|lol|haha|ok|okay|cool|nice)\W*$", 2.0),
        (r"\b(?:who are you|what is your name|tell me about yourself)\b", 1.5),
    ],
    "qa": [
        (r"\?\s*$", 1.0),
        (r"^\W*(?:what|who|whom|whose|which|where|when|why|how|is|are|does|do|did|can|should)\b", 0.75),
        (r"\b(?:explain|describe|define|tell me|what is|what are)\b", 0.75),
    ],
    "rag_qa": [
        (r"\b(?:based on|according to|using|from|in) (?:the |this |these )?(?:context|passages?|documents?|"
         r"articles?|text|sources?|information|above|following)\b", 2.5),
        (r"\b(?:cite|citations?|provided (?:text|context|documents?|passages?))\b", 1.0),
    ],
}

#: Cues that a task needs several steps / careful reasoning (complexity signal).
MULTISTEP_CUES = (
    r"\bstep[- ]by[- ]step\b",
    r"\b(?:first|then|next|finally|afterwards)\b",
    r"\b(?:explain why|justify|reason about|walk me through)\b",
    r"\b(?:compare|contrast|trade-?offs?|pros and cons|evaluate|critique)\b",
    r"\b(?:prove|derive|optimi[sz]e)\b",
)
