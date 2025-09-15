"""
metrics.py — Métricas operables del Parser

Objetivo:
- Medir éxito de parseo por lote (%docs_ok).
- Estimar 'pérdida de layout' como proxy rápido (bloques 'unknown' / total).
- Consistencia de tablas vs. un valor 'golden' (si se conoce para ciertos docs).

Buenas prácticas:
- Reporta estas métricas en dashboards (APM) y en el pipeline de CI.
- Usa un 'golden set' pequeño para validar regresiones (tabla/heading counts).
"""

from __future__ import annotations
from typing import List, Optional, Dict
from schemas import DocumentIR

def percent_docs_ok(results: List[Optional[DocumentIR]]) -> float:
    """
    % de documentos parseados exitosamente en un lote.
    - Considera None como fallo (excepciones).
    """
    total = len(results)
    ok = sum(1 for r in results if r is not None)
    return (ok / total) * 100 if total else 0.0

def layout_loss(doc: DocumentIR) -> float:
    """
    Proxy de 'pérdida de layout':
    - Proporción de bloques 'unknown' vs total de bloques.
    - En este MVP retornará 0 salvo que el parser clasifique 'unknown'.
    - Útil para detectar cuando el parser pierde estructura (p.ej. PDFs escaneados).
    """
    total = 0
    unknown = 0
    for p in doc.pages:
        for b in p.blocks:
            total += 1
            if b.get("type") == "unknown":
                unknown += 1
    return (unknown / total) if total else 0.0

def table_consistency(doc: DocumentIR, golden_tables_per_doc: Optional[int] = None) -> Dict[str, float]:
    """
    Consistencia de tablas:
    - Cuenta tablas encontradas y compara con un valor esperado (si existe).
    - Útil en 'golden sets' donde sabemos cuántas tablas deberían detectarse.
    """
    found = 0
    for p in doc.pages:
        for b in p.blocks:
            if b.get("type") == "table":
                found += 1
    if not golden_tables_per_doc:
        # Sin referencia, devolvemos ratio=1.0 por convención (no penaliza)
        return {"found": float(found), "expected": 0.0, "ratio": 1.0}
    return {
        "found": float(found),
        "expected": float(golden_tables_per_doc),
        "ratio": float(found) / float(golden_tables_per_doc) if golden_tables_per_doc else 1.0,
    }
