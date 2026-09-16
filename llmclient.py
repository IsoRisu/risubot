# main.py
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
import urllib.error
import urllib.request

from botkey import GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

from freeflow_llm import FreeFlowClient

TEXT_MODEL = "gemini-3.6-flash"
VISION_MODEL = "gemini-3.6-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Gemini accepts images, audio, video and PDF as inline data.
SUPPORTED_PREFIXES = ("image/", "audio/", "video/", "application/pdf")


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


def _gemini_generate(prompt: str, attachments: list, model: str) -> str:
    """Single direct call to Gemini's generateContent endpoint."""
    parts = [a.to_part() for a in attachments]
    if prompt:
        parts.append({"text": prompt})

    body = json.dumps({"contents": [{"role": "user", "parts": parts}]}).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from None

    candidates = payload.get("candidates") or []
    if not candidates:
        # Usually a safety block or an empty response.
        raise RuntimeError(f"No candidates returned: {json.dumps(payload)[:500]}")

    chunks = [p["text"] for p in candidates[0]["content"]["parts"] if "text" in p]
    return "".join(chunks).strip()


def ask(prompt: str, attachments=None) -> str:
    """
    Send a prompt, optionally with attachments (list of Attachment), and
    return the reply as a string.

    With no attachments this behaves exactly as before and keeps FreeFlow's
    provider fallback. With attachments it bypasses FreeFlow.
    """
    attachments = attachments or []

    for a in attachments:
        if not a.mime_type.startswith(SUPPORTED_PREFIXES):
            raise ValueError(f"Unsupported attachment type: {a.mime_type}")

    if attachments:
        return _gemini_generate(prompt, attachments, VISION_MODEL)

    with FreeFlowClient() as client:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=TEXT_MODEL,
        )
        return response.content


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