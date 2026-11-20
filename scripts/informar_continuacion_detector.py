"""Verifica linaje y registra la ampliación del mismo entrenamiento desde cero."""
import hashlib
import json
from pathlib import Path

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    historico=(RAIZ/'documentacion/10_representacion_espacial_e_instancias.md').exists()
    nombre='detector_mapas_v6_auxiliar_ext48'
    d=leer(f'resultados/{nombre}.json');anterior=leer('resultados/detector_mapas_v6_auxiliar.json')
    base=leer('resultados/detector_mapas_poligonos_v2.json')
    if d['estado']!='piloto_completado':raise ValueError('Esperar al fin de la continuación')
    if d['historial'][:len(anterior['historial'])]!=anterior['historial']:raise ValueError('Historia inicial distinta')
    if d['hashes_manifiestos']!=anterior['hashes_manifiestos']:raise ValueError('Datos cambiados')
    origen=d['origen_continuacion']
    for campo,sha in [('checkpoint','sha256'),('mejor_checkpoint','sha256_mejor')]:
        path=Path(origen[campo]);path=path if path.is_absolute() else RAIZ/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=origen[sha]:raise ValueError('Checkpoint de origen alterado')
    config=leer('configuracion/arquitectura.json');actual=config['detector']
    if d['val']['map50']>actual['map50_validacion']+1e-8:
        actual.update(checkpoint=d['checkpoint_mejor'],experimento_seleccionado=nombre,
            map50_validacion=d['val']['map50'],map50_95_validacion=d['val']['map50_95'],
            parametros_implementados=d['parametros'],formulacion='softmax7',auxiliar_clasificacion=True,
            modo_inferencia='mapas_directos',informe='documentacion/09_continuacion_entrenamiento_conjunto.md')
    actual.update(informe_ultimo_experimento='documentacion/09_continuacion_entrenamiento_conjunto.md',
        estado='experimental_validacion_insuficiente',test_evaluado=False)
    if not historico:(RAIZ/'configuracion/arquitectura.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    nuevas=[h for h in d['historial'] if h['epoca']>=origen['epoca_inicio']]
    l=['# Continuación del entrenamiento conjunto','',
        f"**Estado: mejor AP50 val registrado {actual['map50_validacion']:.3%}; calidad insuficiente para presentar detección fiable.** No se evalúa test.",'',
        '## Motivo y protocolo','',
        'El piloto con clasificación auxiliar alcanzó su mejor AP50 en la última época de su presupuesto inicial (16). Su pérdida seguía descendiendo. Se amplió el mismo aprendizaje, manteniendo extractor, cabezas, pérdidas, datos, aumentos, lote, learning rate y optimizador. No se importa un modelo ni se inicia otro con pesos ajenos.','',
        'La continuación comienza en la época 17 con tope total 48, mínimo total 32 y paciencia diez por AP50 val. El calentamiento de dos épocas ya estaba cumplido: no se reinicia. El presupuesto y la parada son los únicos cambios de configuración permitidos, además del nombre del archivo de salida.','',
        'Se restauran pesos propios, optimizador AdamW, scaler, RNG CPU/CUDA, generador del cargador de detección y generador del muestreo auxiliar. Se guarda otra pareja de checkpoints y se conserva el mejor estado anterior si ninguna época nueva lo supera. Los originales no se sobrescriben.','',
        f"El origen es {origen['checkpoint']}; SHA256 `{origen['sha256']}`. El mejor origen es {origen['mejor_checkpoint']}; SHA256 `{origen['sha256_mejor']}`. Los hashes se comprobaron después del entrenamiento y siguen coincidiendo.",'',
        'Las primeras 16 entradas del historial se verificaron idénticas a las del piloto original. Los hashes de manifiestos TACO, contornos, inventario, fotos auxiliares RealWaste y caché auxiliar coinciden exactamente. Se continúa con las mismas 964 escenas TACO train y 2.714 fotos RealWaste train. Las fotos globales no añaden cajas o máscaras de detección.','',
        '## Resultados acumulados','',
        '| Ejecución | Épocas acumuladas | Mejor época | mAP@50 | mAP@50:95 | Recall@50 |',
        '|---|---:|---:|---:|---:|---:|']
    for titulo,x in [('Base de contornos sin auxiliar',base),('Piloto conjunto original',anterior),('Continuación del mismo piloto',d)]:
        m=x['val'];l.append(f"| {titulo} | {x['epocas_ejecutadas']} | {x['mejor_epoca']} | {m['map50']:.3%} | {m['map50_95']:.3%} | {m['recall50']:.1%} |")
    l+=['',f"Se añadieron {len(nuevas)} épocas; las épocas iniciales no se cuentan como una ejecución independiente. El aumento frente al piloto conjunto de 16 épocas es de {(d['val']['map50']-anterior['val']['map50'])*100:.3f} puntos porcentuales en AP50. Se utiliza como predeterminado {actual['checkpoint']}.",'',
        'La evaluación conserva las mismas 181 imágenes completas, 361 objetos, seis materiales, regiones ignoradas y evaluador propio de AP interpolado. Umbral de evaluación 0,05, mínimo dos celdas por componente y filtrado del padding previos a formar regiones. No se eliminan objetos pequeños ni se seleccionan imágenes por el acierto.','',
        '| Material | Objetos val | AP50 a 16 épocas | AP50 continuación |',
        '|---|---:|---:|---:|']
    for c,m in d['val']['por_clase'].items():l.append(f"| {c} | {m['objetos']} | {anterior['val']['por_clase'][c]['ap50']:.3%} | {m['ap50']:.3%} |")
    l+=['',f'![Curvas acumuladas; línea discontinua marca la continuación](../resultados/figuras/{nombre}.png)','',
        f'![Clasificación auxiliar de train](../resultados/figuras/{nombre}_auxiliar.png)','',
        f'![Primeras escenas completas val](../resultados/figuras/{nombre}_comparacion_val.png)','',
        '## Error en la consulta por imagen','',
        'En la primera imagen val fijada, taco:686, el modelo propone una caja de plástico demasiado amplia y dos de metal sobre partes del objeto. Con umbral de visualización 0,35, ninguna de las tres predicciones coincide con material e IoU≥0,5: tres falsos positivos y dos objetos omitidos. El resultado se conserva en resultados/ejemplo_continuacion_val_686.json y la conexión al asistente en resultados/integracion_continuacion_val_686.json. El ejemplo no se sustituye por otro favorable.','',
        'Esta imagen empeora respecto al piloto anterior aunque el AP50 medio de toda val aumente. La limitación debe tenerse en cuenta antes de presentar una demostración.','',
        '## Diagnóstico semántico de train','',
        'Se midieron las mismas 964 escenas train y 60 recortes estratificados. Resultados completos en resultados/ajuste_semantico_continuacion_train.json. El diagnóstico utiliza etiquetas GT para medir; no representa precisión de prueba ni AP de cajas.','',
        '## Límites','',
        'El AP50 es el criterio de selección y no implica que todas las métricas o clases mejoren. La accuracy auxiliar sigue siendo una medición de clasificación en train aumentado, no calidad de detección. Los resultados de validación se han utilizado repetidamente para desarrollar el método; no son una prueba final independiente ni demuestran significación estadística.','',
        'La detección orgánica sigue contando con cinco instancias train y una val; las fotos de clasificación global no sustituyen anotaciones de instancia. Los componentes pueden unir objetos próximos y perder detalles pequeños. El video fiable, los FPS, las escenas vacías y la evaluación externa siguen pendientes.','',
        'El siguiente cambio debe centrarse en la representación espacial y la localización: seguir añadiendo épocas con este mismo protocolo dejó de mejorar tras la época 19. No se presenta la ampliación del presupuesto como solución completa.','',
        '## Comprobaciones y reproducción','',
        'Pasaron 59 pruebas, incluidas las que impiden cambiar datos, optimización o peso auxiliar al continuar, reutilizar el nombre de origen o importar pesos externos por este camino. Se verifican ambas fuentes de datos y ambos generadores aleatorios.','',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco -m unittest discover -s pruebas',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_auxiliar_ext48.json --continuar-desde checkpoints/detector_mapas_v6_auxiliar_ultimo.pt',
        '```','',
        'Para reanudar una interrupción del mismo experimento se usa --reanudar con su archivo _ultimo.pt y la misma configuración. Para repetir se necesita otro nombre; la continuación existente y los archivos de origen se protegen contra sobrescritura.','']
    (RAIZ/'documentacion/09_continuacion_entrenamiento_conjunto.md').write_text('\n'.join(l),encoding='utf-8')
    readme=RAIZ/'README.md';p=readme.read_text(encoding='utf-8').split('\n\n')
    for i,t in enumerate(p):
        if t.startswith('**Detector:**'):
            valor=f"{actual['map50_validacion']:.2%}".replace('.',',')
            p[i]=f"**Detector:** Transformer experimental con tarea auxiliar de clasificación, entrenado desde cero y continuado con trazabilidad. Mejor mAP@50 val: **{valor}**; calidad insuficiente. Pasaron 59 pruebas. Historial, comparación por material y límites en [documentacion/09_continuacion_entrenamiento_conjunto.md](documentacion/09_continuacion_entrenamiento_conjunto.md). Test reservado."
    if not historico:readme.write_text('\n\n'.join(p),encoding='utf-8')
    print(json.dumps({'seleccionado':actual['experimento_seleccionado'],'map50_val':actual['map50_validacion'],
        'epocas_acumuladas':d['epocas_ejecutadas'],'mejor_epoca':d['mejor_epoca'],'test_evaluado':False},indent=2))


if __name__=='__main__':main()
