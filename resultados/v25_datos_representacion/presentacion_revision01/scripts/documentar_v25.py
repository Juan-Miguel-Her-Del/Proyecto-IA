"""Informe V25 trazable a JSON; no selecciona modelos ni entrena."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import CLASES
from scripts.capturar_v25 import SALIDA,leer


def mdtabla(cabecera,filas):
    return '\n'.join(['| '+' | '.join(map(str,cabecera))+' |','| '+' | '.join(['---']*len(cabecera))+' |',
        *['| '+' | '.join(map(str,f))+' |' for f in filas]])


def main():
    c=leer(SALIDA/'censo/resumen.json')['splits'];a=leer(SALIDA/'analisis/resumen.json')['splits']
    o=leer(SALIDA/'contrafactuales_resumen.json');d=leer(SALIDA/'distribuciones.json');p=SALIDA
    trozos=['''# V25: diagnóstico de datos y representación

**Decisión: V20 queda cerrado y no adoptado. La siguiente intervención prioritaria es una colección real de detección, diversa por escena y material, con anotación visible por instancia y adjudicación de solapes.** No iniciar otra pérdida ni entrenamiento. La arquitectura semántica tiene un límite real de separación que habrá que abordar después; cambiarla ahora sin resolver cobertura y ambigüedad de supervisión no está justificado por una atribución causal aislada.

Se mantiene el checkpoint original de época 19, `detector_mapas_v6_auxiliar_ext48_mejor.pt`, SHA256 `eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9`, argmax de siete clases, cinco vistas, umbral 0,05, área mínima 2, C8, NMS 0,5 y top100. Visualización 0,35. No cambian pesos, pérdidas, arquitectura, entrenamiento ni manifiestos; test continúa reservado.

## Alcance y método

Se censaron **todas las 3.267 anotaciones de las 964 escenas train**: 2.505 positivas de seis materiales y 762 ignoradas; además, 586 anotaciones de val, 361 positivas y 225 ignoradas. Cada imagen se abrió y comprobó por dimensiones, SHA256 de archivo y de píxeles RGB. Cada polígono se vinculó al índice interno, ID original, categoría, caja y geometría originales mediante `ContornosTACO`; las fusiones de imágenes duplicadas requieren procedencia documentada. Se comprobó también la correspondencia completa con `configuracion/datasets.json`. No se fabricaron positivos ni cajas a partir de RealWaste.

La inferencia nueva recibe únicamente píxeles y metadatos autorizados —ID, ruta, dimensiones y SHA— de las escenas **train completas**, nunca recortes escogidos por GT. Se conserva una vista (lotes de 8) y cinco vistas (completa más cuatro recortes fijos, lote de 5). AMP FP16, modelo eval y gradientes deshabilitados. El observador V19 exige igualdad exacta con el postprocesamiento de producción sobre los mismos logits. Pesos antes/después idénticos y ningún gradiente.

La configuración numérica coincide con la referencia histórica V22: SDPA disponible histórico, sin imponer MATH V23, determinismo estricto/cuDNN determinista desactivados, cuDNN benchmark=False y TF32 histórico. Se registran las opciones efectivas. **Val no tuvo nueva inferencia**: reutiliza la captura histórica exacta V24, cotejada con las predicciones históricas de una y cinco vistas; sus causas se reutilizan de V19 con comprobación de igualdad de predicciones. Todas las consultas a GT se realizan después, offline.

Se mantiene el orden de causas de V19, sin ajustar umbrales a val: correcto, pérdida de postprocesamiento, fusión, fragmentación, material incorrecto, ausencia foreground, geometría insuficiente y soporte GT ambiguo. Es una atribución operacional ordenada, no una demostración causal: los indicadores de fusión/fragmentación también se guardan sin exclusión mutua.

## Datos, procedencia y tamaño
''']
    filas=[]
    for cl in CLASES:
        x=c['train']['censo']['por_material'][cl];y=c['val']['censo']['por_material'][cl]
        filas.append([cl,x['instancias'],x['escenas'],x['elegibles'],x['tamanos'].get('menos_8_px',0),x['tocan_borde'],y['instancias']])
    trozos.append(mdtabla(['Material','GT train','Escenas train','Regiones elegibles','Lado <8 px','Toca borde','GT val'],filas))
    trozos.append('''

![Balance de materiales](../resultados/v25_datos_representacion/figuras/balance.png)

Plástico representa 65,07 % de las instancias positivas; la relación plástico/orgánico es 326:1. El vidrio tiene 203 instancias pero solo 62 escenas: **53 en taco:1321, 26 en taco:598, 26 en taco:1365, 14 en taco:637 y 6 en taco:1335**, 61,58 % del vidrio en cinco escenas. Hay 134 fragmentos, 65 botellas, 3 vasos y 1 tarro. Los conteos de fragmentos no equivalen a diversidad visual independiente. La concentración por escena se conserva para las seis clases en `concentracion.json`; su descriptor de Kish no se presenta como tamaño muestral inferencial.

El cartón tiene 100/149 objetos de lado menor ≥32 píxeles en la entrada; 8/149 son <8. El vidrio tiene 107/203 <8, pero también 38 ≥32. Globalmente hay 502/2.505 objetos train <8 (20,04 %), frente a 73/361 val (20,22 %). Un token corresponde a 8×8 píxeles de la entrada letterbox, no a 8 píxeles de la fotografía original. Una región elegible no equivale a un objeto detectable ni garantiza que tenga dos celdas conectadas.

### Calidad estructural y revisión de etiquetas

No hubo fallos de lectura, hashes, correspondencia de procedencia ni de mapeo de clase; no hubo polígonos de área geométrica nula. Todas las cajas/polígonos tienen IoU entre sus envolventes ≥0,9. Eso comprueba coherencia geométrica, **no corrección del material ni exhaustividad de la anotación**.

Train contiene 44 anotaciones con cruces estrictos de aristas (43 positivas y 1 ignorada); 4 con coordenadas originales fuera de imagen (3 positivas y 1 ignorada). Se conservan las coordenadas y el tratamiento histórico; un cruce es un caso de revisión, no una etiqueta falsa demostrada. Hay 16 parejas positivas con IoU de cajas ≥0,5, 10 del mismo material. En val hay 6 cruces y ninguna coordenada fuera de imagen. Las instancias de borde son candidatos a truncamiento, no una medición directa de visibilidad parcial.

Las hojas de contacto incluyen **todas las positivas**, ID, categoría TACO, contorno y tamaño. Se inspeccionaron las 149 de cartón, 203 de vidrio y 5 orgánicas de train; también las 25 de cartón y 22 de vidrio de val. Es una revisión visual exploratoria de las clases prioritarias. Los otros materiales tienen censo estructural exhaustivo, indicadores por instancia y hojas completas, pero **no adjudicación visual exhaustiva**. No se certifica el material de todos los objetos ni se estima una tasa de etiquetas erróneas. La cola CSV cubre todas las positivas e identifica qué requiere adjudicación humana. Ninguna etiqueta cambió.

Ejemplos a revisar: cartón impreso/brillante con ventanas plásticas; objetos finos/oscuros (taco:1179/3649); partes de botella ocluida de taco:1335/4255–4260, donde debe aclararse si son instancias distintas o partes de un objeto. El aspecto RGB de una botella transparente no prueba su composición física. Las categorías excluidas, como Drink carton y Carded blister pack, mantienen su condición ignorada; no se reasignan por apariencia.

La política histórica permite que positivos poligonales prevalezcan sobre zonas previamente ignoradas por cajas: se registran 1.220 celdas train y 267 val en la unión ignore/padding antes de esa asignación. No se interpreta automáticamente como defecto: preservar positivos frente a cajas ignore amplias puede ser intencional. El censo conserva también los positivos fuera de contenido por centro antes del recorte del contrafactual. **No se corrige silenciosamente esta política.**

### Los cinco orgánicos

![Contexto y detalle de todos los orgánicos train](../resultados/v25_datos_representacion/organicos_contexto.jpg)

| Imagen / ID | Contenido visible | Tokens con mezcla / tocados | Consecuencia regional |
|---|---|---|---|
| taco:68 / 163 | Comida dentro de lata | 23/23, metal | Sin soporte no ambiguo |
| taco:72 / 171 | Pan bajo envoltorio transparente | 187/187, plástico | Sin soporte no ambiguo |
| taco:81 / 203 | Resto pequeño sobre caja | 4/4, cartón | Sin soporte no ambiguo |
| taco:81 / 204 | Otro resto pequeño sobre caja | 4/4, cartón | Sin soporte no ambiguo |
| taco:92 / 244 | Resto de manzana sostenido por una mano | 0/76 | 40 tokens interiores elegibles |

En la entrada rasterizada 384, la fracción del polígono orgánico superpuesta con otro material es 100 %, 99,98 %, 100 % y 100 % en los cuatro primeros casos. La política de un material por token borra esa supervisión compartida. No son solo dos objetos pequeños: **también se excluyen los dos orgánicos grandes**. Aumentar a 96 no recupera ninguno de esos cuatro en el mapa conjunto. La única instancia elegible se reconoce en train (1 TP), lo cual no demuestra generalización: val tiene un solo orgánico pequeño y 0 TP. Su AP train de 10,396 % no tiene respaldo para una conclusión robusta.

## Detector original sobre escenas train completas
''')
    filas=[]
    for s in ['train','val']:
        for v in ['1','5']:
            m=a[s]['metricas'][v]['resumen'];filas.append([s,v,*[f'{100*m[k]:.4f}' for k in ['map50','map50_95','recall50','precision50']],m['tp50'],m['fp50'],m['fn50']])
    trozos.append(mdtabla(['Split','Vistas','mAP50 %','mAP50:95 %','Recall %','Precisión %','TP','FP','FN'],filas))
    trozos.append('\n\n'+mdtabla(['Split / vistas',*CLASES],[[s+' / '+v,*[f"{100*a[s]['metricas'][v]['resumen']['ap50_por_material'][cl]:.4f}" for cl in CLASES]] for s in ['train','val'] for v in ['1','5']]))
    trozos.append('''

El evaluador propio usa AP de 101 puntos e ignore regions; no se declara equivalencia con COCOeval. Train **no es evaluación independiente** y no se usa para adoptar un checkpoint. No se comparan estas escenas completas con el antiguo diagnóstico de memorización de seis escenas como si fueran el mismo conjunto. El dato decisivo es que el detector ya tiene recall 23,71 % en train, y vidrio solo 1 TP/203 con cinco vistas. No basta explicar el resultado por un cambio visual entre train y val.

### Causas principales en cinco vistas
''')
    filas=[]
    campos=['correcto','material_incorrecto','ausencia_respuesta_fg','geometria_insuficiente','fusion_objetos','fragmentacion','perdida_postprocesamiento','sin_soporte_gt_no_ambiguo']
    for s in ['train','val']:
        for cl in ['total',*CLASES]:
            q=a[s]['propuestas']['total'] if cl=='total' else a[s]['propuestas']['por_material'][cl]
            filas.append([s,cl,q['gt'],*[q['causas'][k] for k in campos]])
    trozos.append(mdtabla(['Split','Material','GT','TP','Material','Fondo/ausencia','Geometría','Fusión','Fragmentación','Filtros','Ambiguo'],filas))
    trozos.append('''

![Causas train y val](../resultados/v25_datos_representacion/figuras/errores_train_val.png)

En train, 981 fallos principales son material/ausencia y 829 geometría/fusión/fragmentación; solo 65 se atribuyen primero al postprocesamiento, de 1.911 FN. En val: 176, 91 y 14 respectivamente, de 287 FN. La regla ordenada puede asignar fusión antes que material; no son recuentos causales independientes.

Las oportunidades de cartón train son 24 GT con alguna caja del material correcto e IoU≥0,5 antes de filtros y 19 después; vidrio 1→1. En val ambos permanecen 0→0. En total train son 665→600 oportunidades frente a 594 TP: una misma propuesta puede cubrir más de un GT, mientras el evaluador exige emparejamiento uno a uno. Por eso oportunidades y TP no son intercambiables.

### Cabeza principal: material y fondo por instancia

Se reutilizan logits de la vista completa y las regiones interiores/fallback del protocolo V20. Se excluyen tokens ambiguos, padding e ignore. No interviene `clasificador_auxiliar`. La matriz conserva voto de siete clases y material condicionado (se ignora fondo **solo para el diagnóstico**). La recuperación exige ≥50 % de masa regional correcta; el argmax de votos puede diferir por empates o pluralidad. Esto tampoco es AP ni inferencia usando GT.
''')
    filas=[]
    for s in ['train','val']:
        for cl in CLASES:
            q=a[s]['principal']['por_material'][cl];filas.append([s,cl,*[q[k] for k in ['elegibles','recuperadas','condicionales','pred_fondo','pred_material_erroneo']]])
    trozos.append(mdtabla(['Split','Material','Elegibles','Recuperación ≥50%','Condicional ≥50%','Voto fondo','Voto otro material'],filas))
    trozos.append('''

![Matrices de la cabeza principal](../resultados/v25_datos_representacion/figuras/confusiones_principales.png)

En el voto de siete clases, cartón train→plástico=54 y →metal=13 (67/146), frente a 11 y 4 en val (15/24). Condicionando solo a material: 86 y 13 en train, 16 y 4 en val. Vidrio tiene 159/201 votos a fondo en train y 16/22 en val. La recuperación condicional train de vidrio es 67/201, pero solo 3/201 alcanzan la recuperación con fondo incluido. Se observan tanto problemas de material como de foreground; el buen clasificador global no soluciona automáticamente la cabeza principal.

## ¿Qué impide representar el mapa 48×48?

Contrafactual **offline GT**, no ejecutado como detector: se rasterizan polígonos a entrada 384 fija, se cambia solo la rejilla de análisis a 48/96/384, y se extraen C8 con mínimo dos celdas y padding por centro. Se calcula la mejor IoU de cualquier componente del material frente a cada caja GT; también la mejor caja si se conserva la máscara individual de cada objeto antes de unir materiales. No hay scores, NMS ni emparejamiento uno a uno. Las propuestas pueden reutilizarse entre GT.

La rejilla más fina altera cuantización, conectividad y el área física del filtro de dos celdas. **No añade detalle RGB** y no predice el resultado de entrenar un mapa 96. La rasterización inicial sigue limitada a 384. Las máscaras individuales usan GT conocido: **no son una arquitectura implementada, una mejora real ni un techo absoluto de AP**. Un modelo podría dibujar separaciones distintas del GT. La comparación revela pérdidas de representación, no cuánto aprendería otra cabeza.
''')
    filas=[]
    for s in ['train','val']:
        for n in ['48','96','384']:
            q=o[s][n]['total'];filas.append([s,n,q['instancias'],q['caja_semantica_iou50'],q['caja_por_instancia_iou50'],q['fusion_mayoritaria'],q['sin_token_correcto']])
    trozos.append(mdtabla(['Split','Rejilla','GT','Caja mapa IoU50','Caja por instancia IoU50','Fusión ≥50%','Sin token material'],filas))
    trozos.append('''

![Contrafactual de geometría](../resultados/v25_datos_representacion/figuras/oraculo_geometria.png)

En train, 48→96 recupera 331 oportunidades geométricas y pierde 7 (saldo 324); desaparecen 123 fusiones mayoritarias. A 384 persisten 793 instancias en componentes compartidos mayoritarios: **aumentar resolución no conserva por sí mismo identidades de objetos en contacto o solapados**. En 48, conservar identidad individual sube las cajas posibles de 1.140 a 1.576, con las mismas imágenes y polígonos. La coincidencia de 1.070 instancias con contacto de componente y 17 fragmentadas reproduce el diagnóstico histórico del informe 10, distinguiendo contacto de la regla más estricta de ≥50 % (1.047).

Para cartón train, el mapa GT 48 ofrece 98/149 cajas IoU50; con máscaras individuales 131/149. En val son 18/25 y 22/25. Para vidrio train son 52/203 y 69/203, frente a 141 y 190 a rejilla 384. Val vidrio: 11/22 y 12/22 en 48, 19 y 22 en 384. Existen limitaciones de tamaño y de identidad, pero **el cero de cartón/vidrio val tampoco es impuesto por la geometría de todos sus GT**: muchas cajas serían representables y el modelo no produce el material.

Los errores por tamaño y por material están completos en `tablas/tamanos_y_errores.csv`. Las cinco vistas aumentan el lado menor de 190 objetos train y 30 val desde <8 hasta ≥8 en alguna vista útil, sin producir una solución global: también agregan FP y pueden perder cajas en los bordes. No se ajustó ningún recorte, umbral o regla.

## Generalización visual: evidencia y límites
''')
    trozos.append(mdtabla(['Material','Mediana lado train','Mediana lado val','Categorías train / val'],[[cl,f"{d['por_material'][cl]['mediana_lado_menor_384']['train']:.2f}",f"{d['por_material'][cl]['mediana_lado_menor_384']['val']:.2f}",str(len(d['por_material'][cl]['categorias']['train']))+' / '+str(len(d['por_material'][cl]['categorias']['val']))] for cl in CLASES]))
    trozos.append('''

Train usa diez lotes TACO, val solo batch_2 y batch_9, sin lotes comunes. No hay categorías originales val ausentes de train, pero cambia mucho su composición: vidrio train es 66 % fragmentos y val **22/22 botellas**; su lado mediano cambia de 7,29 a 19,00. Cartón train tiene cajas grandes/corrugadas y escenas repetidas de recipientes; val incluye impresos, superficies deterioradas y objetos parciales. Son diferencias descriptivas, no verificación de independencia por objeto físico ni prueba causal de domain shift.

Se guardan brillo, saturación, varianza de Laplaciano e histogramas RGB por imagen y resúmenes por escenas de cada material. Estos atributos incluyen fondo y la varianza mezcla textura con desenfoque; no certifican blur ni explican por sí solos el error. La revisión de contactos aporta ejemplos verificables; no se deriva de estos descriptores una nueva pérdida ni umbral.

Hay un deterioro de cartón entre train (19 TP/149, AP50=5,638 %) y val (0/25), compatible con generalización insuficiente. En vidrio el fallo ya existe en train y en objetos representables, lo que exige considerar ajuste/aprendizaje, escasez/diversidad y supervisión. Sin intervenir una variable no se puede repartir causalmente el error entre datos, optimización y arquitectura. Esta auditoría no demuestra que todos los problemas sean de datos ni que una cabeza nueva los resolvería.

## Revisión de experimentos previos: qué no repetir

| Evidencia histórica | Resultado y limitación |
|---|---|
| Informe 8, presencia/material y recortes | V5 0,495 % mAP50; recortes balanceados 0,613 %; reclasificación de regiones 0,394 %. No rescatar estas formulaciones como novedad; algunos protocolos agrupan cambios. |
| Informe 10, consultas | V8 cubría 15,6 % del GT train con alguna caja IoU≥0,5; V10 30,5 %, material de mejores cajas 61,2 %, argmax nulo 99,8 %. Mejor caja reutilizable no es AP; modelos/épocas no aíslan arquitectura. |
| Informe 11, centros | Capacidad GT con cajas perfectas conserva 2.496/2.505 tras NMS; hay 22 colisiones centro/celda con máximo 2. Rendimiento real val: centros 2,455 % una vista y 2,114 % cinco, frente a mapas 2,446 % y 2,912 %. Capacidad geométrica no demuestra aprendizaje. |
| Informe 13, fondo | Fondo=1 reduce material→fondo pero aumenta FP y no mejora globalmente. Mantener fondo=3. |
| Informes 17–18, regional auxiliar | Señal sobre cabeza auxiliar no resolvió AP50 cartón; B no adoptada. Una sola región orgánica concentra supervisión. |
| Informe 19, propuestas | Cartón/vidrio val tienen 0 oportunidades antes de filtros; top100 no era el límite dominante. |
| Informes 20–24, margen en cabeza principal | A/B completos y reproducibles V23; B final 1,5013 % y mejor 2,4283 % mAP50 cinco vistas, cartón/vidrio AP50=0. Inferior al original 2,9123 %. V20 cerrado. |

Son ensayos de desarrollo sobre una sola validación reutilizada, diferentes presupuestos/inicializaciones y una sola semilla; no constituyen una comparación causal común de familias arquitectónicas. No se reentrenó ninguno durante V25.

## Intervención elegida y propuesta concreta de datos

**Primero construir una colección por instancia con diversidad física y espacial y supervisión visible coherente.** Tiene prioridad sobre otra pérdida y sobre reentrenar centros/consultas: corrige la única región orgánica, la concentración de vidrio, el sesgo de materiales y la falta de una prueba externa. No se promete una ganancia de mAP sin medirla. La evidencia también justifica rediseñar posteriormente la separación de instancias; ese trabajo es una segunda etapa, con los datos auditados como requisito.

Propuesta operacional inicial, todavía **no recolectada ni anotada**:

| Material | GT train actual | Nuevas instancias reales propuestas | Total orientativo |
|---|---:|---:|---:|
| plástico | 1.630 | 150 | 1.780 |
| papel | 109 | 300 | 409 |
| cartón | 149 | 300 | 449 |
| vidrio | 203 | 250 | 453 |
| metal | 409 | 150 | 559 |
| orgánico | 5 | 400 | 405 |

Son **1.550 nuevas instancias en unas 500–650 escenas reales**. El criterio es superar aproximadamente 400 instancias por clase minoritaria sin inflar conteos con fragmentos de una foto; reduce el cociente máximo/mínimo orientativo de 326 a 4,4. No es un tamaño óptimo estimado ni un cálculo de potencia de AP. Exigir ≥100 escenas y ≥100 objetos físicos distintos en cada clase prioritaria (orgánico, vidrio, cartón, papel), máximo cinco instancias por escena para cumplir su cuota y ≥300 regiones orgánicas sin ambigüedad válidas a 48. No basta recolectar 400 fotos del mismo objeto.

Priorizar orgánicos visibles fuera de envoltorios, distintas frutas/restos/comida y deterioro; vidrio de botellas, vasos y tarros además de fragmentos, transparencia y reflejos; cartón liso/corrugado/impreso, sucio, deformado y con ventanas identificadas. Añadir pares reales de vidrio/PET y cartón/metal o envases brillantes bajo fondos e iluminación comparables. Fotografiar interiores/exteriores, sombras/contraluz, fondos de mesa y calle, distancias que cubran <8, 8–16, 16–32 y ≥32 píxeles a entrada 384, objetos aislados y próximos/solapados. Guardar fotogramas de MP4/webcam con secuencias agrupadas, nunca como ejemplos independientes por frame. No manipular ni recrear residuos peligrosos para aumentar categorías.

Anotar una identidad por objeto, polígono visible y bbox derivada, sin extender máscaras a través de oclusores. Definir explícitamente transparencia, contenido frente a recipiente, huecos y multipartes; el formato TACO de listas de polígonos no expresa huecos sustractivos y debe documentarse antes de importar anotaciones nuevas. Los objetos compuestos o no identificables se etiquetan como tales/ignore, sin forzar una de seis clases. Identificar material por procedencia física cuando la imagen no lo resuelva. Registrar oclusión, truncamiento, fuente, licencia/permiso, objeto físico, sitio/día/dispositivo/secuencia. Doble anotación de todas las prioritarias y 20 % de las demás; adjudicar desacuerdos, registrar acuerdo de material, IoU entre anotadores, omisiones y cobertura regional. Las revisiones de TACO se guardarían en una versión nueva, sin editar los manifiestos históricos. RealWaste mantiene solo sus etiquetas globales.

### Colección nueva para comprobar generalización

Separar **120–180 escenas adicionales**, con ≥100 instancias y ≥50 escenas por material, agrupadas por objeto físico, secuencia, sesión y día/sitio, sin compartir esos grupos con el nuevo train. Capturar también dispositivos/sitios no usados para entrenamiento. Registrar la asignación antes de mirar resultados de modelos. Mantener esta colección bloqueada; no convertirla en otra val usada para elegir pérdidas, umbrales o épocas.

El objetivo de 100 instancias por clase permitiría, bajo independencia ideal, un error máximo aproximado de recall de ±9,8 puntos al 95 %. Como hay agrupamiento por escena, **no se promete esa precisión**: usar intervalos por bootstrap de escenas y documentar soporte. AP tiene otra incertidumbre. Definir previamente criterio conjunto de AP por clase, mAP, recall, FP por escena, geometría y costes. La colección nueva no se evalúa ahora; el test original sigue intacto y reservado. La val actual queda identificada como desarrollo, sin seguir buscando formulaciones en ella.

### Puerta de decisión futura

1. Completar/adjudicar datos y congelar grupos; verificar que las clases prioritarias tienen supervisión visible útil y diversidad, no solo conteos.
2. Antes de gastar GPU, fijar un único protocolo e intervención. No modificar simultáneamente pérdidas, arquitectura y datos para atribuir una mejora.
3. Si se investiga después una cabeza por instancia, exigir conservación de IDs en escenas cercanas y diagnóstico de ajuste sobre train completo. No repetir V11/consultas con otro nombre. El contrafactual prueba capacidad geométrica, no que esa cabeza aprenderá bien.
4. Usar la nueva colección bloqueada solo en la comparación previamente especificada. Nunca adoptar por mejora aislada del clasificador condicional.

## Ejemplos auditables

Ver [índice con IDs y causas](../resultados/v25_datos_representacion/ejemplos/indice.json). Verde: GT analizado; naranja: cajas finales congeladas con score≥0,05, texto para ≥0,35. Los paneles se eligen determinísticamente entre causas existentes y objetos grandes; **no son una muestra aleatoria ni estiman frecuencias**. Las tablas contienen todo el conjunto.

![Cartón train, fallo de material](../resultados/v25_datos_representacion/ejemplos/train_carton_material_incorrecto.jpg)
![Vidrio train, ausencia](../resultados/v25_datos_representacion/ejemplos/train_vidrio_ausencia_respuesta_fg.jpg)
![Cartón val, fallo de material](../resultados/v25_datos_representacion/ejemplos/val_carton_material_incorrecto.jpg)

## Artefactos, costes y reproducción

Directorio: `resultados/v25_datos_representacion/`. `censo/instancias_train.json` incluye positivas e ignoradas; las hojas de `censo/contactos/` incluyen todas las positivas. `tablas/` contiene censo, cola de revisión, métricas, causas, tamaño y contrafactuales. `captura_train/` conserva las predicciones por imagen, logits de una/cinco vistas y trazas V19 con origen de propuestas. `analisis/` conserva errores por GT, métricas completas y confusiones de la cabeza principal. `distribuciones.json`, `organicos_solapes.json`, `concentracion.json`, `propuesta_coleccion.json` y `revision_visual.json` distinguen mediciones, propuesta y triage.

Captura train: una vista **2,99 s**, cinco vistas **267,78 s**, incluyendo archivos y observador, no solo kernels GPU. Pico asignado GPU 79,75 MiB y 54,57 MiB respectivamente (reservado 110 MiB); no son benchmarks de latencia de la app. CUDA, Torch, GPU y backend están registrados en `captura_train/captura.json`. Son 5.784 vistas procesadas en total; cero actualizaciones de optimizador. Los análisis y figuras son CPU.

Pasaron las **193 pruebas existentes y 8 nuevas (201)**. Las nuevas cubren la barrera GT, padding, material ambiguo, objetos pequeños, fusión, separación a distinta rejilla y geometría. La verificación final comprueba hashes de captura/código, IDs/orden, coherencia con predicciones, regresión histórica, censos y preservación de archivos; registra también el manifiesto de integridad V25. Las incidencias de presentación están registradas y corregidas sin repetir inferencia ni modificar históricos.

No se modificó código protegido de entrenamiento, modelo o evaluación; no hay que regenerar ni invalidar certificados V20–V24. V25 tiene huellas propias. El único archivo previo actualizado es el estado del README; sus bytes anteriores se conservan y verifican en `README_previo.md`. Configuraciones originales, métricas y documentos históricos conservan SHA; el checkpoint original se comprueba antes/después. El hash de test se usa solo para preservación de bytes: no se parsean sus anotaciones ni se ejecuta su inferencia.

```powershell
$pyEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pyEco -m unittest discover -s pruebas
& $pyEco scripts/verificar_v25.py
```

Para una repetición independiente, usar otro checkout/directorio de resultados vacío: `capturar_v25.py --preparar`, `auditar_v25.py --fase censo`, `auditar_v25.py --fase analisis`, `informar_v25.py`, `completar_v25.py`, `documentar_v25.py` y `verificar_v25.py --publicar`; generar figuras con `graficar_v25.py` en el intérprete que tenga `requirements-graficas.txt`. Los scripts rechazan sobrescritura de evidencia. No ejecutar otra captura sobre la salida actual.

## Limitaciones metodológicas

Un solo checkpoint y una semilla histórica; train contiene datos vistos por el aprendizaje y val se ha usado repetidamente. El censo automático no descubre de forma fiable todas las instancias omitidas ni verifica composición física. La revisión visual no sustituye adjudicación humana exhaustiva. Los diagnósticos usan polígonos con sus propias limitaciones, criterio de visibilidad de V19 y categorías amplias. Las fracciones por tamaño no controlan simultáneamente material, sitio, oclusión o densidad. Los contrafactuales de resolución y máscaras individuales no son resultados de modelos nuevos. No hay experimento causal que cuantifique cuánto AP ganará cada intervención. La decisión prioriza corregir carencias observadas que limitan cualquier comparación futura, conservando la evidencia de que la separación de instancias también necesitará trabajo.
''')
    out=RAIZ/'documentacion/25_diagnostico_datos_y_representacion.md'
    with out.open('x',encoding='utf-8') as f:f.write('\n\n'.join(trozos))


if __name__=='__main__':main()
