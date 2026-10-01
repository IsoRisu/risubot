# llmclient.py
# Text-and-image in / text out.
#
# Text-only prompts go through freeflow-llm (provider fallback, key rotation).
# Prompts with attachments go straight to the Gemini REST API, because
# freeflow-llm's documented chat() takes string content only, and its fallback
# could otherwise hand an image request to a text-only Groq model.
#
# Import `ask()` from your Discord bot.

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request

from botkey import GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

from freeflow_llm import FreeFlowClient

# Tried in order. When the first is overloaded (503) or rate-limited (429),
# the next one usually still has capacity; they are separate capacity pools.
MODEL_CHAIN = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
TEXT_MODEL = MODEL_CHAIN[0]  # used for FreeFlow's last-resort attempt
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Tells the model it is talking through Discord and exactly how math is shown.
# Keep this in sync with latexrenderer.py: the "supported" list below is what
# matplotlib's mathtext can actually draw (checked against matplotlib 3.10).
SYSTEM_PROMPT = r"""You are a helpful assistant replying in a Discord chat. Reply in the same language the user writes in (Finnish if they write Finnish).

HOW MATH IS DISPLAYED
Discord cannot typeset LaTeX. A renderer on the bot turns math into images, but only math written inside delimiters:
- Inline math: $...$   (use for short expressions inside a sentence)
- Display math: $$...$$   (use for any longer, standalone, or important formula; put it on its own lines)
Everything outside the delimiters is shown as ordinary Discord Markdown (**bold**, *italic*, numbered lists, short headings are all fine).

RULES
1. Every formula, symbol, variable with an index or exponent, and number-with-math goes inside $...$ or $$...$$. Never write bare LaTeX like \frac{1}{2} or x^2 in normal text.
2. Never put math inside code blocks or backticks. Code blocks are only for real source code.
3. When the user asks for math, exercises or formulas "in LaTeX" ("latex-muodossa"), they want them typeset and readable in the chat, NOT a LaTeX source file. Write normal Markdown (numbered list, short intro) with the formulas in $...$ / $$...$$. Never output \documentclass, \usepackage, \begin{document}, \section, \title, \maketitle, itemize/enumerate, or theorem environments. Only if the user explicitly asks for the LaTeX source code / a .tex file, give it inside a ```latex code block.
4. Put words inside math in \text{...}, for example $x \ge 0 \text{ kun } n \to \infty$. Do not wrap whole sentences in $...$.
5. Use $ only for math. Write currency as "5 €" or "5 dollars", never "$5".
6. Multi-line derivations: ONE $$...$$ block, rows separated by \\ , alignment marks & are allowed. Do not use \begin{align}, \begin{equation} or \[ \] unless inside $$...$$.
7. Piecewise functions: use \begin{cases} a & \text{if } x>0 \\ b & \text{otherwise} \end{cases} inside $$...$$.
8. Keep each $$ block short (one idea, one to four rows). Prefer several small blocks over one huge one.

SUPPORTED MATH (use only these; unsupported commands fail and show as raw source)
Fractions and roots: \frac{a}{b}, \sqrt{x}, \sqrt[n]{x}, \binom{n}{k}
Big operators: \sum_{k=1}^{n}, \prod, \int_a^b, \oint, \lim_{x \to 0}, \sup, \inf, \max, \min, \limsup, \liminf
Functions: \sin \cos \tan \arctan \ln \log \exp
Greek letters: \alpha \beta \gamma \delta \varepsilon \epsilon \theta \lambda \mu \pi \sigma \phi \varphi \omega \Delta \Omega ...
Relations: = \neq \le \ge \leq \geq < > \approx \sim \equiv \in \notin \subset \subseteq \supset \supseteq \mid \to \Rightarrow \Leftrightarrow \iff \implies \mapsto
Logic and sets: \forall \exists \neg \wedge \vee \cup \cap \setminus \emptyset \mathbb{R} \mathbb{N} \mathbb{Z} \mathbb{Q} \mathbb{C}
Other: \infty \partial \nabla \cdot \times \pm \ldots \cdots \to, ^{ } and _{ } for powers and indices, |x|, \|x\|, \left( \right), \lfloor \rfloor, \lceil \rceil, \overline{x}, \hat{x}, \vec{v}, f'(x), \mathrm{d}x, \mathbf{x}, \mathcal{F}, \text{...}, \, \quad
DO NOT USE (not supported): \begin{matrix}/pmatrix/bmatrix/array/tabular, \xrightarrow, \underbrace, \overbrace, \boxed, \cancel, \color, \tag, \label, \newcommand, \usepackage, \begin{align} outside $$, tikz, \displaystyle. For a matrix, write its rows as separate lines of text instead.

GRAPHS OF FUNCTIONS, CURVES AND REGIONS
You cannot draw images, but the bot can draw graphs for you. Put a plot tag on its own line:
[[plot: f(x)]]    or    [[plot: f(x); g(x) from A to B]]
- Inside the tag use plain calculator syntax, NOT LaTeX: x^2, 2*x+1, sin(x), cos(x), tan(x), sqrt(x), abs(x), exp(x), ln(x), log10(x), pi, e.
- Graph of a function of x: write just the expression, e.g. [[plot: x^2 - 2*x]] (or "y = x^2 - 2*x").
- Curve that is not a function of x (circle, ellipse, hyperbola, any equation in x and y): write the equation with a single "=", e.g. [[plot: x^2 + y^2 = 1]], [[plot: x^2/4 + y^2 = 1]], [[plot: x*y = 1]], [[plot: (x-1)^2 + (y+2)^2 = 9]]. The variables are x and y only.
- Inequality (shaded region): write it with <, >, <= or >= (ASCII, not ≤ ≥), e.g. [[plot: x^2 + y^2 < 1]], [[plot: y > x^2]], [[plot: x + y <= 2; x > 0]]. The boundary is drawn dashed for < and >, solid for <= and >=; with several inequalities the shaded regions overlap, and the darker overlap is the solution of the system. Use this when asked to illustrate the solution set of an inequality or system of inequalities. Only one comparison per inequality (no a < x < b; write it as two: x > a; x < b).
- Separate up to 6 graphs with ;  The range "from A to B" is optional and applies to BOTH axes. For functions the default is -10 to 10, so choose a range that shows the interesting behaviour, e.g. [[plot: sin(x); cos(x) from -6.3 to 6.3]]. For curves in x and y, omit the range: the view fits automatically.
- Use a tag whenever the user asks to draw, plot or sketch a function or curve, or when a graph clearly helps an explanation (at most two per reply). Also state the function or equation in the text as math, e.g. "Yksikköympyrä $x^2 + y^2 = 1$:" followed by [[plot: x^2 + y^2 = 1]].
- Never draw graphs as ASCII art, and never put plot tags inside code blocks. (Function plots use [[plot: ...]]; the diagram tags below are for structures.)

DIAGRAMS: GRAPHS, AUTOMATA, TURING MACHINES, TREES
The bot can also draw graph-theory graphs, finite automata, Turing machines, pushdown automata and trees. Put a tag on its own line. The tag name is graph, tree, dfa, nfa, tm or pda; the content is a list of statements separated by newlines or ;
- Edges: "A -- B" (undirected), "A -> B" (directed), "A <-> B" (both ways), "A -> B : 5" (label/weight). For automata put the label between the dashes: "q0 -a-> q1", "q0 -a,b-> q0" (a self-loop is fine). Turing machine rule "read/write,move": "q0 -1/0,R-> q1" (blank is _).
- Other statements: "start: q0", "accept: q2, q3" (double circle), "nodes: x, y" (isolated nodes), "highlight: q1" (colours nodes, e.g. a path or the current state), "title: ...", "layout: circle|spring|lr|tb|tree" (optional; sensible default per tag).
- Turing machine tape snapshot: "tape: 1 0 1 1 _", "head: 2" (0-based cell under the head), "state: q1".
- Node names are single words (letters, digits, _): q0, q_accept, A, 3. Write epsilon as eps. No spaces inside names, no LaTeX, no $ in tags.
Examples:
[[dfa: start: q0; accept: q1; q0 -0-> q0; q0 -1-> q1; q1 -0,1-> q1]]
[[graph: A -- B : 3; B -- C : 1; A -- C : 7]]
[[tree: root -> L; root -> R; L -> LL; L -> LR]]
[[tm: start: q0; accept: qacc; q0 -1/1,R-> q0; q0 -_/_,L-> qacc]]
- Use a tag whenever the user asks to draw, show or illustrate such a structure, or when it clearly helps. At most two diagram tags per reply (plot tags count too). State what the diagram shows in the text, e.g. the language an automaton accepts.
- Never draw these as ASCII art, and never output Graphviz/DOT, Mermaid or TikZ code instead of a tag. Do not put diagram tags inside code blocks.

STYLE
Be concise and clear. Discord messages are limited, so avoid long preambles. For exercises, give a numbered list with each task stated precisely, and add hints or solutions only if asked.
"""

# Gemini accepts images, audio, video and PDF as inline data.
SUPPORTED_PREFIXES = ("image/", "audio/", "video/", "application/pdf")

# Transient HTTP statuses worth retrying (rate limit / overloaded).
RETRY_STATUSES = {429, 500, 502, 503, 504}
SKIP_MODEL_STATUSES = RETRY_STATUSES | {404}  # 404: model id retired/unknown
ATTEMPTS_PER_MODEL = 2
REQUEST_TIMEOUT = 60


class GeminiUnavailable(RuntimeError):
    """Gemini is overloaded / rate-limited / unreachable (worth trying elsewhere)."""


class Attachment:
    """One piece of non-text input: raw bytes plus its MIME type."""

    def __init__(self, data: bytes, mime_type: str, name: str = ""):
        self.data = data
        self.mime_type = mime_type
        self.name = name

    @classmethod
    def from_path(cls, path: str) -> "Attachment":
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"Could not guess a MIME type for {path!r}")
        with open(path, "rb") as f:
            return cls(f.read(), mime_type, os.path.basename(path))

    @classmethod
    def from_bytes(cls, data: bytes, filename: str) -> "Attachment":
        mime_type, _ = mimetypes.guess_type(filename)
        return cls(data, mime_type or "application/octet-stream", filename)

    def to_part(self) -> dict:
        return {
            "inline_data": {
                "mime_type": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }


def _gemini_call(prompt: str, attachments: list, model: str) -> str:
    """One model, with a short retry on transient errors."""
    parts = [a.to_part() for a in attachments]
    if prompt:
        parts.append({"text": prompt})

    body = json.dumps({
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
    }).encode("utf-8")

    def make_request():
        return urllib.request.Request(
            GEMINI_URL.format(model=model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
            method="POST",
        )

    payload = None
    last_err = ""
    for attempt in range(1, ATTEMPTS_PER_MODEL + 1):
        try:
            with urllib.request.urlopen(make_request(), timeout=REQUEST_TIMEOUT) as resp:
                payload = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {text}"
            if e.code not in SKIP_MODEL_STATUSES:
                raise RuntimeError(f"Gemini {model} {last_err}") from None  # e.g. bad key / bad request
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"network: {e}"
        if attempt < ATTEMPTS_PER_MODEL:
            time.sleep(1.5 * attempt)
    else:
        raise GeminiUnavailable(f"{model}: {last_err}")

    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"No candidates returned (blockReason={reason}): {json.dumps(payload)[:300]}")

    candidate = candidates[0]
    parts_out = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts_out if "text" in p).strip()
    if not text:
        raise RuntimeError(f"Empty response (finishReason={candidate.get('finishReason')})")
    return text


def _gemini_generate(prompt: str, attachments: list) -> str:
    """Walk MODEL_CHAIN until one model answers."""
    errors = []
    for model in MODEL_CHAIN:
        try:
            return _gemini_call(prompt, attachments, model)
        except GeminiUnavailable as e:
            errors.append(str(e))
    raise GeminiUnavailable("All Gemini models unavailable: " + " | ".join(errors))


def ask(prompt: str, attachments=None) -> str:
    """
    Send a prompt, optionally with attachments (list of Attachment), and
    return the reply as a string.

    Tries each model in MODEL_CHAIN; text-only prompts fall back to FreeFlow
    as a last resort. Attachments never go through FreeFlow.
    """
    attachments = attachments or []

    for a in attachments:
        if not a.mime_type.startswith(SUPPORTED_PREFIXES):
            raise ValueError(f"Unsupported attachment type: {a.mime_type}")

    if attachments:
        # Only Gemini can take these; FreeFlow's other providers are text-only.
        return _gemini_generate(prompt, attachments)

    # Text only: direct Gemini with model fallback first, because FreeFlow
    # with just a Gemini key has nothing to fall back to. If Gemini is down
    # entirely, FreeFlow gets a turn (useful if you add Groq/etc. keys).
    try:
        return _gemini_generate(prompt, [])
    except GeminiUnavailable as gemini_err:
        try:
            with FreeFlowClient() as client:
                response = client.chat(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=TEXT_MODEL,
                )
                return (response.content or "").strip()
        except Exception as ff_err:
            raise GeminiUnavailable(
                "The AI is overloaded right now, please try again in a minute. "
                f"({str(gemini_err)[:300]})"
            ) from None


def main():
    """Interactive loop. Prefix a line with `file:<path>` to attach something."""
    print("FreeFlow LLM test. Attach with: file:/path/to/img.png what is this?")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        attachments = []
        while user_input.startswith("file:"):
            token, _, rest = user_input[5:].partition(" ")
            try:
                attachments.append(Attachment.from_path(os.path.expanduser(token)))
            except (OSError, ValueError) as e:
                print(f"[error] {e}\n")
                attachments = None
                break
            user_input = rest.strip()

        if attachments is None:
            continue

        try:
            reply = ask(user_input, attachments)
            print(f"Bot: {reply}\n")
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()