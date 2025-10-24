# -*- coding: utf-8 -*-
"""
llm_client.py
"""
import os
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import random

# Carga .env en import
load_dotenv(override=True)

# Configuración determinista global
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
TOP_P = float(os.getenv("LLM_TOP_P", 0))
SEED = int(os.getenv("LLM_SEED", 42))


def get_client() -> tuple[Any, Dict[str, Any]]:
    """
    Devuelve (client, meta) listo para usar con OpenAI o Azure OpenAI.
    meta: { "provider": "openai"|"azure", "model": "...", "extra": {...} }
    """
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    if use_azure:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        if not (endpoint and api_key and deployment):
            raise RuntimeError("Azure OpenAI mal configurado. Revisa .env")

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        return client, {
            "provider": "azure",
            "model": deployment,   # en Azure, 'model' es el deployment name
            "extra": {"api_version": api_version, "endpoint": endpoint}
        }

    # OpenAI “puro”
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    org_id = os.getenv("OPENAI_ORG_ID", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no presente. Configura tu .env")
    client = OpenAI(api_key=api_key, base_url=base_url, organization=org_id)
    model = os.getenv("MENTIONS_LLM_MODEL", "gpt-4o-mini")
    return client, {"provider": "openai","model": model,"extra": {
        "base_url": base_url,
        "org": org_id,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,}
        }


def with_backoff(fn, *, retries=5, base_delay=0.6, max_delay=8.0):
    """
    Ejecuta fn() con backoff exponencial y jitter.
    """
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # típicos rate limit / overload
            if any(code in str(e).lower() for code in ["429", "rate", "overloaded", "timeout", "temporarily unavailable"]):
                delay = min(max_delay, base_delay * (2 ** i)) + random.uniform(0, 0.4)
                time.sleep(delay)
                continue
            # otros errores: re-lanzar
            raise
    # agotado
    raise last_err
