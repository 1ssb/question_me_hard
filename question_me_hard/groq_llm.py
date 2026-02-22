"""Groq LLM integration for question_me_hard."""

import os
from typing import Optional


def create_groq_llm(api_key: str, model: Optional[str] = None):
    """Create an LLM callable backed by Groq's API.
    
    Args:
        api_key: Groq API key
        model: Model name. If None, uses GROQ_MODEL env var or defaults to 
               llama-3.3-70b-versatile (latest stable).
    
    Returns:
        A callable (str) -> str that queries Groq
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "Groq Python client not installed. Install with: pip install groq"
        )
    
    # Use provided model, env var, or latest default
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    client = Groq(api_key=api_key)
    
    def llm_fn(prompt: str) -> str:
        """Query Groq and return the response."""
        message = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            max_tokens=1024,
        )
        return message.choices[0].message.content
    
    return llm_fn
