"""
Centralized prompt templates for DocuMind AI.

Keeping prompts separate:
- Improves maintainability
- Makes prompt-engineering explicit
- Looks very professional in reviews/interviews
"""

from typing import List, Dict


def build_rag_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Build a strict Retrieval-Augmented Generation prompt.

    The model is:
    - Forced to use ONLY document context
    - Prevented from hallucinating
    - Asked to be concise and clear
    """

    if not contexts:
        return f"""
You are DocuMind AI.

The user asked:
"{query}"

There is NO relevant context from the documents.

Respond with:
"I am not confident based on the provided documents."
""".strip()

    context_block = "\n\n".join(
        [
            f"[Source: {c.get('source', 'Unknown')} | Page: {c.get('page', 'N/A')}]\n{c.get('text')}"
            for c in contexts
        ]
    )

    prompt = f"""
You are DocuMind AI, an intelligent document assistant.

INSTRUCTIONS (VERY IMPORTANT):
- Answer ONLY using the information in the CONTEXT.
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- If the answer is not clearly present, say:
  "I am not confident based on the provided documents."
- Keep the answer concise, clear, and well-structured.

CONTEXT:
{context_block}

QUESTION:
{query}

FINAL ANSWER:
""".strip()

    return prompt
