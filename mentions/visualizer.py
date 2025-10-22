# -*- coding: utf-8 -*-
"""
visualizer.py — Visualizador enriquecido de menciones (resumen + contextual)

🎨 Color → dominio semántico
🏷️ Etiqueta visible → tipo de entidad (type)
📘 Incluye leyenda automática de colores (solo la primera vez)
Compatible con notebooks o exportación a HTML
"""

from IPython.display import display, HTML
import html
import re

# ================================================================
# 🎨 Paleta de colores por dominio
# ================================================================
DOMAIN_COLORS = {
    "legal": "#f4b400",
    "financial": "#0f9d58",
    "medical": "#db4437",
    "geopolitical": "#4285f4",
    "ecommerce": "#9c27b0",
    "reviews_and_opinions": "#ff7043",
    "generic": "#9e9e9e",
    "veterinary": "#8bc34a",
}

# Variable global: control de leyenda
_LEGEND_SHOWN = False


def _color_for_domain(domain: str) -> str:
    """Devuelve color basado en dominio, con fallback genérico."""
    return DOMAIN_COLORS.get((domain or "generic").lower(), "#9e9e9e")


def _build_legend_html() -> str:
    """Crea la leyenda visual de dominios y colores."""
    legend_items = []
    for domain, color in DOMAIN_COLORS.items():
        legend_items.append(f"""
        <span style="
            background-color:{color};
            border-radius:6px;
            padding:2px 6px;
            margin:3px;
            display:inline-block;
            color:white;
            font-weight:600;
            font-size:13px;">
            {domain}
        </span>
        """)
    legend_html = "<div style='margin-bottom:8px;'><b>🎨 Leyenda de colores (por dominio):</b><br>" + "".join(legend_items) + "</div>"
    return legend_html


# ================================================================
# 🧱 VISUALIZADOR RESUMEN (etiquetas por mención)
# ================================================================
def display_mentions(data: dict):
    """Muestra las menciones como etiquetas coloreadas (color=dominio, etiqueta=tipo)."""
    global _LEGEND_SHOWN

    mentions = data.get("mentions", [])
    if not mentions:
        display(HTML("<p style='color:gray'>⚠️ No hay menciones para mostrar.</p>"))
        return

    html_out = ["<div style='font-family:Segoe UI, sans-serif; line-height:1.8em;'>"]

    # Solo mostrar la leyenda la primera vez
    if not _LEGEND_SHOWN:
        html_out.append(_build_legend_html())
        _LEGEND_SHOWN = True

    html_out.append("<h4 style='margin-bottom:0.5em;'>📄 Menciones extraídas:</h4>")

    for m in mentions:
        color = _color_for_domain(m.get("domain"))
        text = html.escape(m.get("text", ""))
        type_label = html.escape(m.get("type", "?")).upper()
        domain_label = html.escape(m.get("domain", "generic")).lower()
        html_out.append(f"""
        <span style="
            background-color:{color};
            border-radius:6px;
            padding:2px 6px;
            margin:2px;
            display:inline-block;
            color:white;
            font-size:14px;
            font-weight:600;"
            title="Dominio: {domain_label}">
            {text} <span style="font-size:11px; opacity:0.8;">{type_label}</span>
        </span>
        """)

    html_out.append("</div>")
    display(HTML("".join(html_out)))


# ================================================================
# 🔍 VISUALIZADOR CONTEXTUAL (resalta en el texto original)
# ================================================================
def highlight_mentions_in_text(doc_ir: dict, mentions: list):
    """
    Renderiza el texto completo del documento con menciones resaltadas.
    🎨 Color → dominio
    🏷️ Etiqueta → tipo de entidad
    📘 Leyenda mostrada solo una vez
    """
    global _LEGEND_SHOWN

    if not mentions:
        msg = "<p style='color:gray'>⚠️ No hay menciones para resaltar.</p>"
        display(HTML(msg))
        return HTML(msg)

    # Extraer texto concatenado del documento IR
    text_blocks = []
    for p in doc_ir.get("pages", []):
        for b in p.get("blocks", []):
            if isinstance(b.get("text"), str):
                text_blocks.append(b["text"].strip())
    full_text = "\n".join(text_blocks)
    if not full_text.strip():
        msg = "<p style='color:gray'>⚠️ No hay texto legible en el documento.</p>"
        display(HTML(msg))
        return HTML(msg)

    safe_text = html.escape(full_text)

    # Reemplazar menciones ordenadas (evita solapamientos)
    for m in sorted(mentions, key=lambda x: len(x.get("text", "")), reverse=True):
        frag = m.get("text", "")
        if not frag:
            continue
        color = _color_for_domain(m.get("domain"))
        type_label = html.escape(m.get("type", "?")).upper()
        domain_label = html.escape(m.get("domain", "generic")).lower()
        t = re.escape(frag)
        span = (
            f"<span style='background-color:{color}; border-radius:6px; padding:1px 4px;"
            f"color:white; font-weight:600;' title='Dominio: {domain_label}'>"
            f"{html.escape(frag)} <sub style='font-size:11px;opacity:0.8'>{type_label}</sub></span>"
        )
        safe_text = re.sub(t, span, safe_text, count=1, flags=re.IGNORECASE)

    html_out = ["<div style='font-family:Segoe UI, sans-serif; line-height:1.7em; white-space:pre-wrap;'>"]

    # Leyenda solo la primera vez
    if not _LEGEND_SHOWN:
        html_out.append(_build_legend_html())
        _LEGEND_SHOWN = True

    html_out.append(f"""
        <h4>📜 Texto con menciones resaltadas:</h4>
        {safe_text}
    </div>
    """)
    display(HTML("".join(html_out)))
    return HTML("".join(html_out))
