"""Cuantifica el fallo de material frente a geometría, sin inferencia nueva."""
import hashlib
import json
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CL=['plastico','papel','carton','vidrio','metal','organico']


def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pct(a,b):return f'{100*a/b:.2f} %' if b else 'sin soporte'


def main():
    met=leer(OUT/'metricas_piloto.json');resultados={}
    for n in ['Original','A','B']:
        resultados[n]={}
        for dom in ['historico_train','nuevo_train','desarrollo','externo_retenido']:
            a=leer(RAIZ/met['resultados'][n][dom]['analisis']);h=a['cabeza_principal']
            cm=h['matriz_instancias_con_fondo'];cc=h.get('matriz_instancias_condicional_material',h.get('matriz_instancias_material_condicional'))
            filas=[]
            for i,c in enumerate(CL):
                total=sum(cm[i+1]);cond=sum(cc[i])
                if total!=cond:raise ValueError('Soporte regional incoherente')
                filas.append({'material':c,'regiones_con_tokens':total,'material_correcto_incluye_fondo':cm[i+1][i+1],
                    'voto_fondo':cm[i+1][0],'material_incorrecto':total-cm[i+1][0]-cm[i+1][i+1],
                    'correcto_condicional_material':cc[i][i],'predicho_plastico_condicional':cc[i][0],
                    'geometria_5v':a['5']['geometria']['por_material'][c]})
            resultados[n][dom]=filas
    visuales=leer(OUT/'visuales_revision02.json')
    for v in visuales:
        if hashlib.sha256((RAIZ/v['archivo']).read_bytes()).hexdigest()!=v['sha256']:raise ValueError('Figura cambiada')
    with (OUT/'resumen_material_fondo_geometria.json').open('x',encoding='utf8') as f:json.dump(resultados,f,ensure_ascii=False,indent=2)
    doc=RAIZ/'documentacion/30_arquitectura_hibrida_cnn_transformer.md'
    with (OUT/'informe_antes_complemento.md').open('xb') as f:f.write(doc.read_bytes())
    texto=['','## Diagnóstico dirigido y siguiente intervención','',
        'La siguiente tabla usa voto mayoritario sobre tokens GT válidos de cada instancia, no cajas finales. El clasificador condicional compara solo los seis materiales y excluye fondo; por ello puede mostrar un error material aunque la predicción final sea fondo. No confundir regiones elegibles con el total de cajas GT.','',
        '| B, train histórico | Regiones con tokens | Acierto principal | Voto fondo | Acierto condicional | Predicho plástico condicional |','|---|---:|---:|---:|---:|---:|']
    for r in resultados['B']['historico_train']:
        total=r['regiones_con_tokens']
        texto.append('| '+' | '.join([r['material'],str(total),pct(r['material_correcto_incluye_fondo'],total),pct(r['voto_fondo'],total),pct(r['correcto_condicional_material'],total),str(r['predicho_plastico_condicional'])])+' |')
    texto+=['',
        'En B histórico, 104/146 cartones se votan como plástico aun excluyendo fondo; solo 36/146 reciben el material correcto condicional. Vidrio tiene 177/202 votos fondo, y solo 5/202 materiales correctos condicionales. Papel no tiene aciertos principales por instancia y solo 2/108 condicionales; metal 42/397 condicionales. El único orgánico elegible se vota como plástico. No hay evidencia de que postprocesamiento o separación de cajas, por sí solos, resuelvan estos errores de material.','',
        '**Intervención prioritaria recomendada:** corregir la cobertura y distribución de supervisión de los materiales difíciles antes de otra pérdida o de entrenar automáticamente la cabeza por consultas. Priorizar detección real anotada de vidrio, restos alimentarios, papel y cartón, con diversidad de escenas y mapeo comprobado; utilizar únicamente auxiliares que pasen su QA y el protocolo autorizado. Un siguiente muestreo balanceado por material debe tratarse como un experimento separado, conservando exposición histórica y controles, no como una corrección silenciosa de V30.','',
        'La cabeza propia por consultas sigue preparada para un estudio de localización individual cuando exista señal material suficiente. Predice cajas desde características CNN+Transformer sin componentes conexos, pero aquí no se entrenó ni se midió su AP. Su inicialización deberá seguir siendo aleatoria propia; no se propone transferir pesos de época 19 sin autorización. Un único piloto no permite separar causalmente escasez de datos, presupuesto insuficiente y decisiones de representación.','',
        '### Exposición orgánica','',
        'El censo V25 conserva cinco cajas Food waste. Cuatro no tenían tokens regionales no ambiguos; la única región interior elegible es taco:92 / objeto 244 (40 tokens con aquel criterio). Su imagen aparece siete veces en el plan histórico de 800 pasos. Los 76 tokens orgánicos de la prueba de memorización usan la máscara semántica de cobertura y no son otra instancia ni contradicen el criterio interior anterior. PackWISE/CLICK aportan cero positivos orgánicos nuevos. No se inventa una descomposición de loss por clase que no fue registrada; se preserva la presencia/exposición comprobada en exposicion_organicos.json.','',
        '### Matices necesarios','',
        '- B histórico produce 615 TP y 8779 FP, frente a 592 TP y 16239 FP del original, pero su mAP macro es inferior y deja materiales completos sin AP. Más TP agregados no demuestra mejora de seis clases. Como A/B parten desde cero, no se denomina “olvido” a su diferencia frente al modelo original más entrenado.','',
        '- B en desarrollo 5v: 84 TP, 1377 FP, 277 FN; AP50 plástico 7,0445 %, metal 0,1100 %, papel/cartón/vidrio/orgánico cero. El resultado no cumple un sistema fiable de separación de residuos.','',
        '- B retenido externo 5v: 8,4623 % mAP50, A 7,1921 %, original 1,6374 %. Solo tres clases con soporte y grupos heurísticos: esto no sustituye la evaluación de seis materiales ni acredita generalización independiente.','',
        '- El benchmark 5v de B es más corto que el original, pero incluye postprocesamiento y B produce otras cantidades de propuestas. No demuestra que la CNN sea intrínsecamente más rápida. En 1v B tarda 29,54 ms frente a 23,00 ms del original; ninguno de esos tiempos incluye cargar/recortar las fotografías de alta resolución ni acredita FPS sostenidos de webcam.','',
        '### Evidencia adicional y visuales revisados','',
        'certificado_inicializacion_desde_cero.json comprueba que los pesos del paso 0 se recrean exactamente con la semilla y AdamW está vacío. certificado_piloto_detenido.json verifica las capturas y la reproducción exacta observada de las predicciones de desarrollo 1v entre la puerta inicial y la auditoría, sin afirmar identidad universal entre backends. entorno_ejecucion.json registra Python y bibliotecas utilizadas.','',
        'La incidencia AMP afectó a siete elementos de tokenizador_detalle.bias a escala 65536, no a una loss no finita. El diagnóstico contrastó los mismos pesos y logits con escalas 1024/FP32; el cambio quedó versionado y todos los controles de la revisión 02 se regeneraron.','',
        'Los primeros 19 paneles se conservaron. La revisión visual 02 corrige la leyenda GT para no aplicarle un umbral de confianza y muestra las cajas ignore en gris. No cambia ningún contorno, etiqueta, caja predicha ni métrica. Índice recomendado: resultados/v30_hibrido_revision02/visuales_revision02.json; las fotografías continúan excluidas de Git.','',
        'Ejemplo prefijado de cartón en train nuevo, con ignore visible; no se presenta como anotación humana revisada del proyecto:','',
        '![GT separado y predicciones](../datos/nuevos/v30/privado/visuales_revision02/07_nuevo_train_carton.jpg)','',
        'Este informe documenta una arquitectura híbrida real y un resultado negativo frente a la referencia seleccionada. No se amplía el presupuesto, no se ejecuta otra pérdida y no se utiliza test.','']
    with doc.open('a',encoding='utf8') as f:f.write('\n'.join(texto))
    with (OUT/'complemento_interpretacion.json').open('x',encoding='utf8') as f:json.dump({'matrices':resultados,'visuales_revision02':visuales,'cambio_hipotesis':False,'entrenamiento_adicional':False},f,ensure_ascii=False,indent=2)
    print('Interpretación añadida y visuales revision02 verificados',flush=True)


if __name__=='__main__':main()
