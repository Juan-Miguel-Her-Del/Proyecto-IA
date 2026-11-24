"""Informe y figuras a partir de evidencia guardada; no carga modelos ni entrena."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ/'resultados/v30_hibrido_revision02'
CLASES = ['plastico', 'papel', 'carton', 'vidrio', 'metal', 'organico']


def leer(p): return json.loads(Path(p).read_text(encoding='utf8'))
def pct(x): return 'sin GT' if x is None else f'{100*x:.3f} %'


def main():
    cfg = leer(RAIZ/'configuracion/entrenamiento_hibrido_v30_revision02.json')
    memo = leer(OUT/'controles/memorizacion.json')
    met = leer(OUT/'metricas.json') if (OUT/'metricas.json').exists() else None
    formas = leer(OUT/'arquitectura_medida.json')
    texto = ['# V30 — CNN + Transformer propio desde cero', '',
        '## Autorización y alcance', '',
        'El usuario comunica autorización docente para CNN y autoriza esta arquitectura híbrida. No se atribuye autorización de pesos preentrenados. A y B se inicializan con pesos aleatorios propios; no reciben pesos ni AdamW de época 19. La referencia original se usa únicamente para comparar. No se evalúa test ni se desbloquea la colección de generalización V26.', '',
        '## Auditoría previa', '',
        'V25 midió 4,224 % mAP50 train y 2,912 % desarrollo; 502/2505 objetos train tienen lado menor a 8 píxeles, vidrio concentra 107/203 instancias bajo ese tamaño. Solo hay cinco orgánicos históricos en cuatro escenas y una región plenamente elegible para el criterio regional anterior. Esto limita cualquier arquitectura. V28.2 mejoró papel/orgánico en auxiliar visto y empeoró el agregado histórico, sin AP auxiliar de vidrio/metal. V29 mejoró el retenido de tres clases de 1,637 % a 6,932 %, principalmente plástico, conservando metal AP cero. No se interpreta como mejora de seis clases.', '',
        'Consultas V8–V10 ya exploraron asignación uno a uno, densidad, posiciones y centros de atención; V11 usó picos/regresión de centros. V30 mide primero el campo receptivo convolucional y características compartidas manteniendo la cabeza de mapas. Además, `DetectorHibridoInstancias` implementa una alternativa funcional propia CNN+Transformer+consultas, atención cruzada con padding excluido, asignación uno a uno, cajas y siete logits (seis materiales/ausencia). Sus pruebas verifican gradientes, vacío y que dos cajas próximas del mismo material no se fusionan por componentes/NMS. Esa alternativa no se entrena en la primera ablación: cambiar simultáneamente extractor y cabeza impediría atribuir el resultado a CNN. No se reporta desempeño de C sin entrenarla.', '',
        '### Modelos, librerías y herramientas históricas', '',
        '| Tipo | Componentes | Uso |', '|---|---|---|',
        '| Modelos propios | CNN residual, bloques Transformer, fusión, cabezas mapas/consultas, clasificador auxiliar y asistente propio | Inferencia/aprendizaje local con pesos propios |',
        '| Librerías matemáticas | PyTorch Conv2d/Linear/GroupNorm/LayerNorm/SDPA/AMP/AdamW, NumPy | Operaciones y diferenciación, sin cargar detectores ajenos |',
        '| E/S y geometría | Pillow, OpenCV, codec COCO, Matplotlib | Leer fotografías/video, máscaras, postprocesar logits y graficar; no reconocer materiales por un modelo ajeno |',
        '| Auxiliares históricas | SAM/SAM 2 y editor de anotación; anotación asistida declarada por autores CLICK | Propuestas y correcciones con procedencia; nunca extractor o detector final |', '',
        'No se descarga ningún modelo de los autores ni se invoca un servicio de reconocimiento. Se distinguen propuestas IA pendientes, correcciones humanas del usuario, anotaciones de autores externos y aceptación V26. La autorización de COCO con licencia procede del alcance dado por el usuario; no se inventa aprobación docente individual de cada dataset.', '',
        '## Arquitectura y control', '',
        '```mermaid', 'flowchart LR', 'RGB["RGB 384×384 + máscara geométrica"] --> CNN["CNN residual propia: 24 / 48 / 96 canales"]',
        'CNN --> D["48×48×96 → proyección → posición → Transformer local"]',
        'CNN --> C["24×24×96 → proyección → posición → Transformer global"]',
        'D --> F["Fusión con compuerta + LayerNorm"]', 'C --> F', 'F --> H["Cabeza original: 7 logits por celda"]',
        'H --> P["Argmax, componentes, bordes, NMS, top100"]', 'P --> R["Cajas + seis materiales + confianza"]', '```', '',
        'Conv2d 3×3 stride 2, GroupNorm y SiLU; bloques residuales en escalas 48 y 24. GroupNorm permite microbatches pequeños, usa estadísticas también en inferencia y no necesita running averages. Sus estadísticas incluyen ceros del padding; no se afirma normalización enmascarada. Referencia: [GroupNorm PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).', '',
        'Se reutilizan Bloque/Atencion propios, posición seno/coseno 2D, atención local de ventanas 8×8, contexto global de 576 tokens, fusión original, cabeza semántica y clasificador auxiliar. A conserva tokenización lineal RGB; B la sustituye por CNN. Ambos reciben la misma nueva máscara geométrica para que padding no sea una variable exclusiva de B. La referencia histórica conserva sus operaciones anteriores. Sin padding, el control coincide con el modelo histórico dentro de tolerancia comprobada.', '',
        'La atención excluye claves padding y anula salidas de padding; la CNN anula entrada y salidas inválidas por escala. Las regiones ignore no se convierten en padding: se excluyen de la pérdida original según sus máscaras. Aumentos horizontales transforman conjuntamente imagen, máscara semántica y máscara geométrica. Flatten usa orden fila/columna y conserva la posición. La cabeza principal usa realmente las salidas Transformer; su ablación cambia los logits y sus gradientes son no nulos.', '',
        '| Modelo | Parámetros | CNN | Forma final |', '|---|---:|---:|---|']
    for n, r in formas.items(): texto.append(f'| {n} | {r["parametros"]} | {r["parametros_cnn"]} | B×7×48×48 |')
    texto += ['', '48×48 conserva el control de resolución frente al detector previo. 2304 tokens globales requerirían 16 veces los pares de atención de 576; por eso el detalle usa ventanas y el contexto 24×24. La CNN amplía el campo receptivo pero no elimina la cuantización de objetos menores a ocho píxeles ni la unión de objetos del mismo material que se toquen en el mapa.', '',
        '## Datos y protocolo fijado', '',
        'Train histórico: 964 escenas, 2505 positivos (1630 plástico, 109 papel, 149 cartón, 203 vidrio, 409 metal, 5 orgánico). Nuevo train congelado V29: 1050 escenas, 4624 positivos (3745 plástico, 623 papel, 6 cartón, 250 metal), 1353 ignore. Retenido externo: 240 escenas, 1090 positivos (887 plástico, 108 papel, 95 metal), 339 ignore. No se introducen las propuestas IA pendientes ni se modifican las revisiones humanas. Las categorías multicapa y ambiguas mantienen ignore.', '',
        'Se heredan fotografías/licencias y grupos de [V29](29_entrenamiento_packwise_click.md): PackWISE v2 y CLICK v2, CC BY 4.0 en fichas oficiales; se conserva la discrepancia textual de licencia CLICK. RLE/huecos/polígonos originales y hashes permanecen intactos. Aceptación formal V26 de esta colección: cero. No se inventan identidades físicas, sesiones ni revisores.', '',
        'A y B: semilla 3001, 384×384, 96 canales, ocho imágenes históricas comunes + dos PackWISE + dos CLICK por actualización; tarea global RealWaste train de 16 imágenes balanceadas. CE foreground + 3×CE fondo + Dice originales, mismos pesos por material calculados únicamente en train histórico, microbatch nuevo con coeficiente 0,5 y tarea global 0,5. No se introduce pérdida discriminativa ni regional V20. El balance por fuente y la exposición histórica idéntica evitan repetir sustitución de V28.2, pero no solucionan el desequilibrio interno ni añaden orgánicos/vidrio nuevos.', '',
        '2400 AdamW fijos por rama desde cero, LR 0,0005, weight decay 0,01, clipping 1, AMP/GradScaler, sin warmup ni selección de mejor época. Son 19200 exposiciones históricas y 9600 nuevas por rama, aproximadamente veinte y nueve recorridos; se justifica un presupuesto mayor que las continuaciones de V29 porque todos los parámetros empiezan aleatorios. Checkpoints 0/200/800/1600/2400, sin ampliación posterior a métricas. Gate train a 200: caída media de loss de al menos 10 % entre primeras/últimas 25 actualizaciones. B tiene más parámetros y cómputo: no es un control de FLOPs iguales ni una comparación de múltiples semillas.', '',
        'Corrección del usuario al reanudar: antes del presupuesto completo se evalúa el piloto 800 A/B una única vez, con una vista en desarrollo sin gradientes. Criterio fijado antes de resultados: B mAP50≥1 % y recall≥5 %. Si falla, no se ejecutan los pasos restantes. Este control de factibilidad no sustituye los requisitos finales de integración, y no convierte la val histórica, ya usada por el proyecto, en evaluación independiente.', '',
        'Backend V23 completo explícito: SDPA MATH, cuDNN determinista, benchmark false, algoritmos deterministas, TF32 false, CUBLAS_WORKSPACE_CONFIG=:4096:8, AdamW y clipping foreach false, fused false. [PyTorch reproducibilidad](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md). No se generaliza identidad entre versiones/hardware.', '',
        '## Controles train-only y corrección numérica', '',
        'La preparación r1 se detuvo en el paso 49 de memorización A por gradientes no finitos. El contrafactual conservó exactamente pesos/entradas: logits y loss finitos, mismos logits AMP y gradientes finitos al reducir escala 65536→1024; FP32 también finito. Se identifica overflow del backward escalado AMP, no evidencia de una pérdida no finita. La revisión 02 fija init_scale=1024 en ambas ramas, manteniendo GradScaler adaptativo, FP16, backend, arquitectura, pérdidas, LR, clipping y presupuesto. Se regeneran pruebas y C1/C2/R. Evidencia r1, diagnóstico, configuración y 72 archivos de código congelado quedan conservados en `resultados/v30_hibrido/`; no se usan sus pesos. La revisión tiene rutas y checkpoints exclusivos.', '',
        'Suite histórica conservada y nuevas pruebas de escalas, gradientes, uso efectivo de atención, padding, orden espacial, cajas, vacío, múltiples objetos, checkpoints y compatibilidad. C1/C2/R: cuatro actualizaciones reales por trayectoria/rama; R guarda/restaura al paso 2, cotejando entradas, logits, losses, gradientes antes/después de clipping, pesos, AdamW, scaler y RNG. rtol=1e-4, atol=1e-5; restauración exacta exigida. Los estados Python y NumPy se guardan además de torch/CUDA. No se afirma igualdad bit a bit si la comparación no la registra.', '',
        'Memorización: ocho escenas train seleccionadas por mínimo ID por material más escenas multinstancia. 240 pasos descartables por rama sin aumentos, con la pérdida original; gates prefijados caída loss ≥35 %, accuracy foreground ≥50 % y accuracy ≥30 % en cada material. Demuestra ajuste semántico en esas escenas, no separación garantizada de instancias ni generalización.', '',
        '| Rama | Control memorización | Reducción loss | Pesos |', '|---|---|---:|---|']
    for n, r in memo['ramas'].items(): texto.append(f'| {n} | {r["aprobada"]} | {pct(r["reduccion_loss"])} | descartados |')
    if met:
        res = met['resultados']; filas = []
        texto += ['', '## Resultados finales', '',
            'Inferencia sin GT; análisis posterior usa anotaciones. Umbral 0,05, argmax, mínimo dos celdas, NMS 0,5, top100, bordes de vistas originales. AP propio de 101 puntos con ignore bbox; no se declara COCOeval oficial. El mAP promedia clases con GT: el retenido externo solo mide tres. Desarrollo histórico se ha usado repetidamente en el proyecto y no es una validación independiente nueva. Ninguna métrica de esos conjuntos determina pasos extra o modifica umbrales.', '',
            '| Dominio | Modelo | Vistas | mAP50 | mAP50:95 | Recall | Precisión | TP | FP | FN |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        dominios = ['historico_train', 'nuevo_train', 'desarrollo', 'externo_retenido']
        for dom in dominios:
            for n in ['Original', 'A', 'B']:
                for v in ['1', '5']:
                    r = res[n][dom]['metricas'][v]
                    filas.append({'dominio': dom, 'modelo': n, 'vistas': v, **{k: r[k] for k in ['map50','map50_95','recall50','precision50','tp50','fp50','fn50']}})
                    texto.append('| '+' | '.join([dom,n,v,*[pct(r[k]) for k in ['map50','map50_95','recall50','precision50']],*[str(r[k]) for k in ['tp50','fp50','fn50']]])+' |')
        with (OUT/'tabla_metricas.csv').open('x', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
        texto += ['', '| Dominio, cinco vistas | Modelo | '+' | '.join(CLASES)+' |', '|---|---|'+'---:|'*6]
        for dom in dominios:
            for n in ['Original', 'A', 'B']:
                texto.append('| '+dom+' | '+n+' | '+' | '.join(pct(res[n][dom]['metricas']['5']['ap50_por_material'][c]) for c in CLASES)+' |')
        figdir = OUT/'graficas'; figdir.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        for ax, dom in zip(axs, dominios):
            ax.bar(['Original', 'A', 'B'], [100*res[n][dom]['metricas']['5']['map50'] for n in ['Original', 'A', 'B']]); ax.set_title(dom); ax.set_ylabel('mAP50 (%)')
        fig.tight_layout(); fig.savefig(figdir/'map50.png', dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        texto += ['', '## Tiempo, memoria y checkpoints', '', '| Rama | Pasos | Tiempo bucles | Pico GPU | SHA256 final |', '|---|---:|---:|---:|---|']
        for rama in ['A', 'B']:
            ends = [leer(p) for p in (OUT/f'entrenamiento/{rama}/intentos').glob('*_fin.json')]
            trs = []
            for p in sorted((OUT/f'entrenamiento/{rama}/trazas').rglob('*.json')): trs.append(leer(p))
            trs.sort(key=lambda r: r['paso'])
            ax.plot([r['paso'] for r in trs], [r['loss_total'] for r in trs], alpha=.65, label=rama)
            cp = leer(OUT/f'entrenamiento/{rama}/confirmaciones/paso_2400.json')
            texto.append(f'| {rama} | {sum(r["actualizaciones"] for r in ends)} | {sum(r["segundos"] for r in ends):.1f} s | {max(r["memoria_gpu_max_bytes"] for r in ends)/1024**2:.1f} MiB | {cp["sha256"]} |')
        ax.legend(); ax.set_xlabel('Actualización'); ax.set_ylabel('Loss original + nuevo + global'); fig.tight_layout(); fig.savefig(figdir/'curvas.png', dpi=150); plt.close(fig)
        texto += ['', '| Modelo | Latencia 1v | Latencia 5v |', '|---|---:|---:|']
        for n in ['Original', 'A', 'B']:
            bench = res[n]['benchmark']; texto.append(f'| {n} | {bench["1"]["ms_media_modelo_y_postproceso"]:.2f} ms | {bench["5"]["ms_media_modelo_y_postproceso"]:.2f} ms |')
        texto += ['', 'Latencias con sincronización CUDA: modelo y postproceso, sin disco/letterbox; doce mediciones después de cuatro calentamientos sobre IDs prefijados. Los tiempos de captura incluyen disco y son distintos del benchmark. Los JSON contienen memoria GPU de evaluación y matrices por celda/instancia, confusión fondo/material y geometría. Fusión/fragmentación son indicadores de cajas, no verificación de identidad física.', '',
            '## Decisión', '', f'Puerta de integración prefijada superada: **{met["puerta_integracion_superada"]}**. Se exige B≥max(A,Original)+0,5 puntos mAP50 desarrollo cinco vistas, recall no inferior, FP≤110 % del original y mAP retenido B≥A. El checkpoint predeterminado permanece original incluso si se habilita B como opción. No se adopta automáticamente.', '',
            'La referencia de época 19 se conserva: resultados train/nuevo/retenido reutilizan capturas V29 verificadas; desarrollo se vuelve a medir con backend V23 explícito. Diferencias frente al histórico 2,912 % pueden deberse al backend y no se presentan como reproducción exacta entre backends. La comparación causal de CNN es A contra B desde cero y una semilla; comparar contra época 19 también difiere en historia de exposición.', '',
            'Si B no supera la puerta, no se usa en la aplicación y no se amplía el entrenamiento. Si mejora materiales pero persisten geometría/fusiones, la siguiente intervención debe separar instancias sobre las características híbridas en un protocolo nuevo, reutilizando atención cruzada/asignación uno a uno; no se atribuye a esta CNN una capacidad que la cabeza semántica no tiene. Vidrio/orgánico siguen requiriendo fotografías reales anotadas y cartón mayor diversidad. No se inicia automáticamente ese experimento.', '',
            '![mAP50](../resultados/v30_hibrido_revision02/graficas/map50.png)', '', '![Curvas](../resultados/v30_hibrido_revision02/graficas/curvas.png)', '',
            '## Ejemplos e integración', '',
            'Los ejemplos se seleccionan antes de inferencia por menor ID con cada material/dominio; GT ocupa un panel separado. Predicciones a 0,35 para legibilidad, métricas a 0,05. Índice `resultados/v30_hibrido_revision02/visuales.json`, fotografías locales en `datos/nuevos/v30/privado/visuales` excluidas de Git mediante una regla nueva cuyo .gitignore previo se conserva. No hay redistribución de imágenes ni selección solo de éxitos.', '',
            '`scripts/demostrar_v30.py` permite foto, MP4, webcam solicitada y asistente con las mismas detecciones. Por defecto usa el detector anterior. `--modelo hibrido` exige que B final evaluada supere la puerta; nunca reemplaza configuracion/arquitectura.json. No se simula una prueba de webcam física. Las salidas mantienen advertencias experimentales, seis materiales y orgánico/inorgánico.', '']
    else:
        texto += ['', '## Entrenamiento detenido', '', 'La puerta de diagnóstico no se superó. No se lanza el presupuesto ampliado ni se fabrican métricas. Consultar `controles/memorizacion.json` y `incidencias`. Se conserva el modelo anterior.']
        if (OUT/'piloto_800/puerta.json').exists():
            gate = leer(OUT/'piloto_800/puerta.json')
            texto += ['', '### Evaluación del piloto en desarrollo sin gradientes', '', '| Rama | mAP50 | Recall | Precisión | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
            for n, r in gate['resultados'].items():
                texto.append('| '+' | '.join([n, pct(r['map50']), pct(r['recall50']), pct(r['precision50']), str(r['tp50']), str(r['fp50']), str(r['fn50'])])+' |')
            texto += ['', f'Puerta prefijada superada: {gate["aprobada"]}. No se amplía el presupuesto para forzar un resultado.']
    texto += ['', '## Reproducción e integridad', '',
        'Scripts: entrenar_v30.py --fase congelar; probar_v30.py; entrenar_v30.py --fase controles; --fase memorizar; --fase entrenar --rama A/B --hasta 200, después --hasta 800; evaluar_v30.py --piloto-800; solo si pasa, entrenar --hasta 2400; evaluar_v30.py; visualizar_v30.py; informar_v30.py; verificar_v30.py. Cada publicación rechaza sobrescrituras. Para una réplica usar nuevas rutas/versionado y las mismas versiones/hashes.', '',
        'Checkpoints exclusivos V30 guardan pesos, AdamW, scaler, RNG torch/CUDA/Python/NumPy, historial, plan, hashes de código/datos y backend; la reanudación rechaza mezclas de rama/protocolo y verifica la cadena de commits/trazas. Los metadatos de época 19 no se heredan en los checkpoints nuevos.', '',
        'Suite y controles: `pruebas.json`, `controles/reanudacion.json`, `controles/memorizacion.json`; integridad final en `integridad.json` con recibo SHA256 separado. Requisitos previos conservados en `requisitos_previos.md`, única actualización histórica deliberada documentada por autorización CNN. Los checkpoints antiguos, revisiones humanas, colecciones y métricas V26–V29 conservan sus bytes. Test no se evalúa; no se desbloquea generalización ni se declara aceptación V26.', '']
    p = RAIZ/'documentacion/30_arquitectura_hibrida_cnn_transformer.md'
    with p.open('x', encoding='utf8') as f: f.write('\n'.join(texto)+'\n')
    print(p, flush=True)


if __name__ == '__main__': main()
