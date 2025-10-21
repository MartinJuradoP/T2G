# 🧭 Diagnóstico General del Modelo

| Métrica | Evaluación |
|----------|-------------|
| **Coverage = 1.00** | 🟢 Ideal: Excelente cobertura (todos los fragmentos contextualizados) |
| **Outlier Rate = 0.00** | 🟢 Ideal: Sin ruido |
| **Topic Size (Median) = 6.51** | ⚪ N/A: Métrica no clasificada |
| **Keyword Diversity (basic) = 0.88** | 🟢 Ideal: Alta diversidad (léxica o semántica) |
| **Entropy = 0.19** | 🔴 Problema: Entropía baja (un tópico domina) |
| **Redundancy = 0.00** | 🟢 Ideal: Tópicos bien diferenciados |
| **Keyword Diversity (ext) = 0.89** | 🟢 Ideal: Alta diversidad (léxica o semántica) |
| **Semantic Variance = 0.06** | 🔴 Fuera de rango: Tópicos demasiado homogéneos o dispersos |
| **Semantic Coherence = 0.36** | 🟢 Óptima: Coherencia interna adecuada |

📊 **Diagnóstico final:**
> “🟢 Buena calidad contextual: el modelo agrupa y diversifica correctamente. La entropía baja indica poca diversidad temática.”



# 🔍 2. Desempeño por Fuente de Documento

| Fuente        |   Contextization_Score |
|:--------------|-----------------------:|
| CV            |                  0.778 |
| Reuters       |                  0.477 |
| Wikipedia     |                  0.812 |
| Yahoo Finance |                  0.58  |
| Yelp          |                  0.787 |

📦 **Interpretación general:**
- Fuentes con mayor `Contextization_Score` indican textos más ricos y narrativos (mayor diversidad).
- Valores bajos reflejan documentos técnicos o con poco contexto semántico (como Reuters o Yahoo Finance).


# 🧩 3. Evaluación por Categoría

| Categoría | Score promedio | Diagnóstico |
|------------|----------------:|--------------|
| **Cobertura y estructura** | 1.00 | 🟢 Ideal |
| **Diversidad y separación** | 0.69 | 🟡 Aceptable |
| **Calidad semántica interna** | 0.21 | 🔴 Problema |

# 🔧 4. Recomendaciones Técnicas

- Incrementar `hybrid_eps` (0.2→0.3) para permitir más clusters y aumentar diversidad temática.
- Probar embeddings más ricos (`all-mpnet-base-v2`) para mayor heterogeneidad semántica.