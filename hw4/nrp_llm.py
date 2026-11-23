"""
Shared connection settings for the NRP Nautilus LLM service (OpenAI-compatible API).

The API key is read from the environment variable OPENAI_API_KEY. A .env file in
this folder or any parent folder is loaded automatically (python-dotenv). The key
is never hard-coded or printed.
"""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

NRP_BASE_URL = os.environ.get("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
CHAT_MODEL = os.environ.get("NRP_CHAT_MODEL", "gpt-oss")            # served as openai/gpt-oss-120b
EMBED_MODEL = os.environ.get("NRP_EMBED_MODEL", "qwen3-embedding")  # 4096-dim embeddings


def _api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file (OPENAI_API_KEY=...) "
            "or export it in your shell."
        )
    return key


def get_chat_model(temperature=0.0, reasoning_effort=None, **kwargs):
    """LangChain chat model for gpt-oss on Nautilus."""
    from langchain_openai import ChatOpenAI

    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=temperature,
        api_key=_api_key(),
        base_url=NRP_BASE_URL,
        timeout=120,
        max_retries=2,
        **kwargs,
    )


def get_embeddings():
    """LangChain embedding model served by Nautilus (no local model download)."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=_api_key(),
        base_url=NRP_BASE_URL,
        # The server is not OpenAI, so send raw strings instead of tiktoken ids
        check_embedding_ctx_length=False,
    )
