"""
Grounded Prompt Formulation Subsystem.

Formulates generation prompts adhering to strict factual grounding contracts:
1. Zero extrapolation: Answers must derive exclusively from provided context blocks.
2. Explicit refusal directive: "I cannot find sufficient information in the provided documents."
3. Citation syntax: Bracketed numeric citations ([1], [2]) directly attached to factual assertions.
"""

from __future__ import annotations

from ragnarok.context.contract import AssembledContextPayload

REFUSAL_MESSAGE = "I cannot find sufficient information in the provided documents."

DEFAULT_GROUNDED_SYSTEM_PROMPT = """You are RAGnarok, an authoritative technical assistant specialized in document-grounded question answering.

Strict Grounding Guidelines:
1. Answer the user question using ONLY the factual information provided in the Context sources below.
2. Never infer, hypothesize, or extrapolate beyond the explicit facts in the context blocks.
3. If the provided context sources do NOT contain sufficient factual evidence to answer the question, you MUST explicitly output:
   "I cannot find sufficient information in the provided documents."
   Do NOT attempt to guess, apologize, or answer from general outside knowledge.
4. Every key factual claim in your answer MUST be followed immediately by its corresponding source citation index, e.g., [1] or [2].
5. Never invent citation numbers (e.g., do not cite [99] if only [1] and [2] are provided).
6. Do not combine citations into hyphenated ranges (e.g., avoid [1-3]; cite explicitly as [1], [2], [3]).
"""


def format_user_prompt(query: str, context_text: str) -> str:
    """
    Formats the user message incorporating context sources and user retrieval query.

    Parameters:
        query: User input retrieval query.
        context_text: Delimited [Source X] context blocks string.

    Returns:
        Formatted user prompt string with clear boundary demarcations.
    """
    clean_context = context_text.strip() if context_text else "[No context documents provided]"
    clean_query = query.strip()

    return f"""Context Sources:
{clean_context}

User Question:
{clean_query}"""


def build_generation_messages(
    query: str,
    context: str | AssembledContextPayload,
    system_prompt: str = DEFAULT_GROUNDED_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """
    Pure procedural function that constructs an OpenAI-compatible message list
    for the generation LLM engine.

    Parameters:
        query: The user query to answer.
        context: Context string or AssembledContextPayload from Context Management.
        system_prompt: Grounded system instructions enforcing refusal and citation rules.

    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    context_text = (
        context.assembled_context_text if isinstance(context, AssembledContextPayload) else context
    )

    user_content = format_user_prompt(query=query, context_text=context_text)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
