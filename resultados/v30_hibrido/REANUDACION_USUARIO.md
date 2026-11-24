# Reanudación V30 y corrección académica

El usuario pidió continuar y precisó las restricciones: CNN propia y Transformer propio, inicialización aleatoria, ningún peso preentrenado externo ni reutilización automática de época 19, sin SAM ni servicios de reconocimiento en inferencia. El estado PAUSA_USUARIO.md se conserva como registro del punto anterior.

Se conservó una copia íntegra de los requisitos antes de incorporar la corrección. No se descargaron modelos. La arquitectura alternativa por instancias reutiliza únicamente código propio de atención cruzada y asignación, no los pesos de las consultas anteriores.

La configuración aún no congelada se ajustó antes de medir resultados: tras memorización train-only y piloto de 200 pasos, avanzar como máximo a 800 y medir desarrollo sin gradientes. Solo si B alcanza mAP50 de 1 % y recall de 5 % en esa evaluación prefijada se autoriza completar el presupuesto máximo de 2400. No ampliar ni ajustar la hipótesis si falla. Test y generalización V26 permanecen reservados.

Las fotografías V30 y visuales de terceros son locales y están excluidas de Git mediante una regla nueva; se conserva el .gitignore anterior.
