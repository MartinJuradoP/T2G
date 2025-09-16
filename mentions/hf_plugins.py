# mentions/hf_plugins.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional

# No importamos torch aquí para evitar conflictos de NumPy cuando no se usa HF.
_TORCH: Optional[Any] = None

def _require_torch():
    """Importa torch de forma perezosa y lo retorna. Lanza error claro si falta."""
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyTorch no está disponible. Instálalo o ejecuta sin --use-transformers.\n"
            "Ejemplo: pip install 'torch'  (elige la build adecuada para tu sistema)"
        ) from e
    _TORCH = torch
    return _TORCH

def _pick_device(requested: str):
    """Selecciona cpu/cuda/mps si están disponibles (Apple Silicon)."""
    torch = _require_torch()
    req = (requested or "cpu").lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class HFRelRanker:
    """
    Wrapper mínimo para un clasificador de relaciones (TextClassification) local.
    - Requiere 'model_path' **en disco** (offline), no descarga de internet.
    - Funciona con modelos binarios (1 logit) y multiclase (id2label en config).
    - API:
        * predict(texts) -> List[float]        # score de "hay relación" en [0,1]
        * predict_multiclass(texts) -> List[Dict[label, prob]]
    """
    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 16):
        # Import perezoso de transformers también:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers no está disponible. Instálalo o ejecuta sin --use-transformers.\n"
                "Ejemplo: pip install transformers"
            ) from e

        torch = _require_torch()

        # Carga estrictamente local
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

        self.device = _pick_device(device)
        self.model.to(self.device)
        self.batch_size = int(batch_size)

        # Descubrimos metadatos del modelo (multiclase)
        cfg = self.model.config
        self.num_labels: int = int(getattr(cfg, "num_labels", 1))
        self.id2label: Dict[int, str] = dict(getattr(cfg, "id2label", {}) or {})
        self.label2id: Dict[str, int] = dict(getattr(cfg, "label2id", {}) or {})

        # Política para "ninguna relación" en multiclase
        self.none_label_name = "no_relation"
        self.none_id: int = self.label2id.get(self.none_label_name, -1)

    def _forward_logits(self, texts: List[str]):
        """Tokeniza en batch y retorna logits (B, C) en device."""
        torch = _require_torch()
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.inference_mode():
            out = self.model(**toks).logits
        return out

    def predict(self, texts: List[str]) -> List[float]:
        """
        Devuelve score ∈ [0,1] por texto.
          - Binario: sigmoid(logit)
          - Multiclase:
              * si hay 'no_relation' en label2id -> 1 - P(no_relation)
              * si no, usa max softmax como confianza general
        """
        torch = _require_torch()
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            logits = self._forward_logits(batch)  # (B, C)
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits).squeeze(-1)  # (B,)
                out.extend(probs.detach().cpu().tolist())
            else:
                sm = torch.softmax(logits, dim=-1)  # (B, C)
                if 0 <= self.none_id < sm.shape[-1]:
                    score = 1.0 - sm[:, self.none_id]        # “hay relación” vs. no_relation
                    out.extend(score.detach().cpu().tolist())
                else:
                    score, _ = sm.max(dim=-1)                 # fallback: confianza = max prob
                    out.extend(score.detach().cpu().tolist())
        return out

    def predict_multiclass(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Devuelve distribución por etiqueta (solo multiclase).
        [{"works_at": 0.6, "title": 0.2, "no_relation": 0.1, ...}, ...]
        """
        if self.num_labels <= 1:
            raise RuntimeError("El modelo no es multiclase (num_labels=1).")
        torch = _require_torch()
        dists: List[Dict[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            logits = self._forward_logits(batch)
            sm = torch.softmax(logits, dim=-1)  # (B, C)
            for row in sm.detach().cpu().tolist():
                dist = {self.id2label.get(j, str(j)): float(p) for j, p in enumerate(row)}
                dists.append(dist)
        return dists
