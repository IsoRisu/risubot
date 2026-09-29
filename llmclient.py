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

# Tells the model it is talking through Discord, where math only shows up if it
# is written in delimiters that latexrenderer.py picks up.
SYSTEM_PROMPT = (
    "You are a Discord bot. Reply in the same language the user writes in. "
    "Discord cannot typeset LaTeX by itself; the bot renders math for you, but only "
    "when it is written as $...$ (inline) or $$...$$ (display). "
    "Whenever the user asks for math, formulas, or exercises 'in LaTeX', write them as "
    "normal Markdown text (numbered lists, short headings) with every formula inside "
    "$...$ or $$...$$. "
    "Do NOT output a complete LaTeX document or preamble (\\documentclass, \\usepackage, "
    "\\begin{document}, \\section, enumerate/itemize) and do not wrap math in code blocks, "
    "unless the user explicitly asks for the LaTeX source code. "
    "Use only simple math commands (\\frac, \\sum, \\int, \\lim, \\sqrt, Greek letters, "
    "\\mathbb, \\mathrm). Put multi-line derivations in one $$...$$ block separated by \\\\. "
    "Keep answers reasonably concise."
)

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