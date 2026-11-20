"""Métricas multiclase a partir de predicciones y etiquetas, sin pesos externos."""
def metricas_clasificacion(predichos, reales, clases):
    matriz = [[0] * len(clases) for _ in clases]
    if len(predichos) != len(reales) or not reales:
        raise ValueError("Se requieren predicciones y etiquetas no vacías de igual longitud")
    for p, r in zip(predichos, reales):
        matriz[r][p] += 1
    detalle = {}
    for i, clase in enumerate(clases):
        tp = matriz[i][i]
        precision = tp / max(sum(f[i] for f in matriz), 1)
        recall = tp / max(sum(matriz[i]), 1)
        detalle[clase] = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "soporte": sum(matriz[i])}
    return {"accuracy": sum(matriz[i][i] for i in range(len(clases))) / len(reales),
        "f1_macro": sum(d["f1"] for d in detalle.values()) / len(clases),
        "por_clase": detalle, "matriz_confusion": matriz, "orden_clases": clases}
