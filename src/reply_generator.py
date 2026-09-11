"""Grounded reply drafting from message + optional retrieved cases."""
from __future__ import annotations

from typing import Literal

from src import llm_client
from src.retrieval import RetrievedCase

DraftMode = Literal["auto", "template", "llm"]

_SYSTEM_PROMPT = (
    "You are an Apple Support agent. Reply in 2-4 short sentences. "
    "Use the historical cases only as guidance. "
    "Never invent order numbers, tracking IDs, or account details. "
    "If those identifiers are not in the customer message or cases, do not mention any."
)


def draft_reply(
    message: str,
    retrieved_cases: list[RetrievedCase] | None = None,
    *,
    mode: DraftMode = "auto",
) -> str:
    """Draft a support reply, optionally grounded on retrieved cases."""
    use_llm = mode == "llm" or (mode == "auto" and llm_client.is_available())
    if use_llm:
        try:
            return llm_client.complete(
                _SYSTEM_PROMPT,
                _build_user_prompt(message, retrieved_cases),
            )
        except RuntimeError:
            # Explicit --draft llm must not silently look like a successful LLM eval.
            if mode == "llm":
                raise
            return _template_reply(message, retrieved_cases)
    return _template_reply(message, retrieved_cases)


def _build_user_prompt(
    message: str,
    retrieved_cases: list[RetrievedCase] | None,
) -> str:
    """Format the user turn for the LLM."""
    lines = [f"Customer message:\n{message}"]
    if retrieved_cases:
        lines.append("\nSimilar historical cases:")
        for index, case in enumerate(retrieved_cases, start=1):
            lines.append(
                f"{index}. Customer: {case.customer_msg}\n"
                f"   Agent: {case.brand_reply}"
            )
    else:
        lines.append("\nNo historical cases provided.")
    lines.append("\nWrite the agent reply only.")
    return "\n".join(lines)


def _template_reply(
    message: str,
    retrieved_cases: list[RetrievedCase] | None,
) -> str:
    """Deterministic offline reply used when no API key is configured."""
    opener = (
        "Thanks for reaching out — happy to help with this."
    )
    if not retrieved_cases:
        return (
            f"{opener} I do not have a close historical match on hand, "
            f"so could you share your device model and the exact steps that lead to: "
            f"\"{message[:120]}\"?"
        )
    top = retrieved_cases[0]
    return (
        f"{opener} In a similar case (\"{top.customer_msg[:80]}\"), "
        f"we advised: \"{top.brand_reply[:160]}\" "
        "If that does not match your setup, reply with your iOS version and we will dig in."
    )
