"""
fastapi/nlp/draft_service.py
------------------------------
LLM-powered draft response generator for support tickets.

Uses OpenAI gpt-4o-mini. Falls back gracefully to category-specific
templates when no API key is available or the call fails.

Prompt engineering principles applied:
  • Professional, concise (3–5 sentences max)
  • References only entities confirmed in the ticket text
  • Asks for missing critical information (order ID, email) when absent
  • Category-specific guidance injected per ticket type
  • JSON output enforced via response_format for reliable parsing
  • Temperature 0.3 for deterministic, professional tone

Output schema:
    {
        "ticket_id":         str | None,
        "category":          str,
        "entities":          list[{label, start, end, text}],
        "draft_response":    str,
        "model":             str,
        "prompt_tokens":     int,
        "completion_tokens": int,
        "error":             str | None
    }
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from nlp.ner_service import extract_entities

# ── Per-category guidance injected into the user prompt ──────────────────────
_CATEGORY_GUIDANCE: Dict[str, str] = {
    "Delivery": (
        "The customer has a delivery issue. Acknowledge the delay or missing parcel, "
        "confirm any order ID mentioned, and advise on next steps (courier trace, etc.). "
        "If no order ID is present, ask for one."
    ),
    "Refund": (
        "The customer is requesting a refund or reporting a billing problem. Acknowledge the "
        "request and reference any order ID or charge mentioned. Do NOT promise a specific "
        "refund timeline. If no order ID is given, ask for it."
    ),
    "Account": (
        "The customer has an account access or profile issue. Acknowledge the problem and "
        "reference any email address mentioned. Do NOT ask for passwords. "
        "If no account identifier is provided, ask for the registered email address."
    ),
    "Product Issue": (
        "The customer is reporting a product defect, mismatch, or missing item. Acknowledge "
        "the specific issue and order ID if present. Offer next steps (replacement, return, "
        "or warranty claim). If no order ID is given, request one."
    ),
    "Other": (
        "The customer has a general enquiry. Answer helpfully and professionally. "
        "If the enquiry requires account or order details you do not have, ask for them politely."
    ),
}

_SYSTEM_PROMPT = """You are a professional customer support agent for an e-commerce company.
Write a first-response draft to the customer support ticket provided.

Rules you MUST follow:
1. Be professional, empathetic, and concise — 3 to 5 sentences maximum.
2. Address the customer by their situation, not by name (you do not have their name).
3. Reference ONLY details explicitly stated in the ticket. Do NOT invent order numbers,
   dates, amounts, product names, or any other specifics not present in the text.
4. If critical information is missing (e.g. order ID for a refund), politely ask for it.
5. Never make specific promises about timelines or outcomes you cannot guarantee.
6. Close with a polite offer to assist further.
7. Do not use placeholders like [ORDER_ID] — use the actual value or omit it.

Respond with ONLY a JSON object (no markdown, no preamble):
{"draft_response": "<your reply here>"}"""


def _build_user_prompt(text: str, category: str, entities: List[Dict]) -> str:
    entity_line = ""
    if entities:
        entity_line = "\nExtracted entities: " + ", ".join(
            f"{e['label']}: {e['text']}" for e in entities
        )
    guidance = _CATEGORY_GUIDANCE.get(category, _CATEGORY_GUIDANCE["Other"])
    return (
        f"Category: {category}\n"
        f"Ticket: {text}"
        f"{entity_line}\n\n"
        f"Guidance: {guidance}"
    )


def generate_draft(
    text: str,
    category: str,
    ticket_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a draft response. Returns structured dict including entities."""
    entities = extract_entities(text)
    result: Dict[str, Any] = {
        "ticket_id":         ticket_id,
        "category":          category,
        "entities":          entities,
        "draft_response":    None,
        "model":             model,
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "error":             None,
    }

    if not _OPENAI_AVAILABLE:
        result["error"] = "openai package not installed — run: pip install openai"
        result["draft_response"] = _template_fallback(category, entities)
        return result

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        result["error"] = "OPENAI_API_KEY not set and no key provided"
        result["draft_response"] = _template_fallback(category, entities)
        return result

    try:
        client   = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(text, category, entities)},
            ],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        result["draft_response"]    = parsed.get("draft_response", "")
        result["prompt_tokens"]     = response.usage.prompt_tokens
        result["completion_tokens"] = response.usage.completion_tokens

    except Exception as e:
        result["error"]          = str(e)
        result["draft_response"] = _template_fallback(category, entities)

    return result


def _template_fallback(category: str, entities: List[Dict]) -> str:
    """Conservative template used when the LLM is unavailable."""
    order_ids = [e["text"] for e in entities if e["label"] == "ORDER_ID"]
    emails    = [e["text"] for e in entities if e["label"] == "EMAIL"]

    order_ref  = f" regarding order {order_ids[0]}" if order_ids else ""
    email_note = f" We have your contact email as {emails[0]}." if emails else ""
    ask_order  = "Please provide your order number so we can locate your purchase. " if not order_ids else ""
    ask_email  = "Could you please provide the email address registered to your account? " if not emails else ""

    templates = {
        "Delivery": (
            f"Thank you for reaching out{order_ref}. We're sorry to hear your parcel hasn't arrived "
            f"as expected and will investigate with our courier team right away.{email_note} "
            f"{ask_order}We'll be in touch as soon as we have an update."
        ),
        "Refund": (
            f"Thank you for contacting us{order_ref}. We sincerely apologise for the inconvenience. "
            f"Our team will review your refund request and follow up with an outcome shortly.{email_note} "
            f"{ask_order}Is there anything else we can help you with in the meantime?"
        ),
        "Account": (
            f"Thank you for getting in touch. We're sorry you're having difficulty accessing your account. "
            f"{ask_email if not emails else f'We can see the registered email is {emails[0]}. '}"
            f"Our team will prioritise resolving this and will follow up shortly. "
            f"Please never share your password with anyone, including our support team."
        ),
        "Product Issue": (
            f"Thank you for letting us know{order_ref}. We're very sorry to hear about the issue "
            f"with your item and would like to make this right.{email_note} "
            f"{ask_order}Could you also share a photo of the issue to help us expedite the resolution?"
        ),
        "Other": (
            f"Thank you for your message. We're happy to help with your enquiry. "
            f"A member of our team will review your request and get back to you as soon as possible. "
            f"Please don't hesitate to include any additional details in your reply."
        ),
    }
    return templates.get(category, templates["Other"])
