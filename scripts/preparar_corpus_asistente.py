"""Corpus inicial asistido por IA; las variantes de una pregunta conservan su grupo."""
import json
from pathlib import Path
import random
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.asistente import MATERIALES, contexto_texto, palabras

# Cada línea base es un grupo. Val/test no se crean transformando el train.
PREGUNTAS = {
"resumen": {
"train": """¿Qué residuos detectaste?
Muéstrame el resumen de la imagen
¿Cuántos objetos hay en el fotograma?
Enumera las detecciones actuales
Dame el conteo por material
¿Qué apareció en el video?
Lista los residuos que encontró el detector
Resume los resultados de la cámara
¿Cuántos residuos identificó el sistema?
Indica el total de detecciones
¿Qué se ve en los resultados?
Cuenta los objetos detectados
Quiero saber cuántas detecciones hay
Describe lo encontrado en esta imagen
Revisa el resumen del fotograma actual
Dime qué materiales aparecen y cuántos hay
¿Cuál es el resultado del análisis?
Presenta el inventario de residuos
¿Hay uno o varios objetos detectados?
Necesito el número total de residuos
¿Puedes listar lo detectado ahora?
Muestra las cantidades encontradas
No quiero recomendaciones, solo el conteo
¿Qué encontró la cámara en esta escena?""",
"val": """Haz una lista de los objetos encontrados
¿Cuántas detecciones aparecen ahora?
Necesito un resumen de lo que detectó la cámara
Cuéntame qué residuos hay en esta toma
Indica las cantidades de cada clase detectada
Solo dime el total de objetos del análisis""",
"test": """Recuenta lo que aparece en el último fotograma
¿Qué ha identificado el detector en esta foto?
Quiero consultar las detecciones del video
Organiza el resultado por tipo y cantidad
¿Cuál es el inventario que obtuviste?
Dame una visión general de los residuos encontrados"""},
"material": {
"train": """¿De qué material es este residuo?
¿Es plástico o vidrio?
Explica la clase predicha
¿Esto es papel o cartón?
¿El objeto es orgánico o inorgánico?
Quiero conocer el material identificado
¿Qué significa la etiqueta metal?
Dime a qué clase pertenece el objeto
¿Cuál es el material de la botella detectada?
¿Cómo clasificaste este residuo?
Aclara qué categoría tiene este objeto
¿Qué diferencia hay entre orgánico e inorgánico?
¿Por qué el cartón aparece como inorgánico?
¿Qué representa la clase orgánico?
¿La lata fue clasificada como metal?
Identifica el material que predice el modelo
¿Qué tipo de residuo es el detectado?
Necesito entender el nombre de la clase
Consulta el material de este objeto
¿Qué etiqueta asignó el sistema?
Quiero la clasificación general del residuo
¿Qué material indica la detección?
No pregunté por la caneca, sino por el material
Explica si el residuo es papel, plástico o metal""",
"val": """¿A qué material corresponde esta predicción?
Explica la categoría orgánica del objeto
¿Qué quiere decir que sea inorgánico?
¿Qué clase obtuvo el residuo de la foto?
Ayúdame a entender la etiqueta de la detección
Consulta si lo detectado es vidrio o plástico""",
"test": """¿Con qué material está etiquetado lo que apareció?
Quisiera conocer la categoría del residuo identificado
¿Este objeto pertenece a los orgánicos?
Interpreta el nombre de material mostrado
Aclara la clasificación que realizó la cámara
¿El sistema dice que esto es cartón?"""},
"separacion": {
"train": """¿En qué caneca deposito esto?
¿Dónde debo botar el residuo?
¿Cuál es el recipiente adecuado?
¿Va al blanco, verde o negro?
Recomienda cómo separar los residuos detectados
¿Dónde pongo el plástico?
¿El orgánico va al recipiente verde?
¿Qué color de bolsa corresponde al papel?
Ayúdame a decidir dónde depositar el metal
¿Se puede reciclar este residuo?
¿Qué hago con el vidrio detectado?
Indica la caneca recomendada para el cartón
¿Lo separo con los aprovechables?
¿Debo echar esto al negro?
Necesito una recomendación de separación
¿En dónde se entrega este desecho?
¿A qué contenedor corresponde la botella?
Orienta la disposición de estos objetos
¿Con qué residuos debo juntar esto?
¿El papel va a la bolsa blanca?
¿Cómo reparto lo detectado entre las canecas?
¿Puedo enviar este material al reciclaje?
No necesito el conteo, dime dónde va
¿Qué recipiente uso para separar este objeto?""",
"val": """¿Qué caneca le corresponde al residuo detectado?
Dime si esto va a reciclaje o a los no aprovechables
Quiero orientación para depositar lo encontrado
¿Cuál de los tres colores debo usar?
¿A dónde llevo el material de esta imagen?
Explica cómo distribuir los residuos en recipientes""",
"test": """¿Debo usar la bolsa verde para este objeto?
Necesito elegir el contenedor para lo detectado
¿Esto se deposita en el recipiente blanco?
Recomiéndame la separación para este material
¿Lo junto con el reciclaje o con la basura ordinaria?
Indica el destino adecuado del desecho identificado"""},
"preparacion": {
"train": """¿Debo lavar el envase antes de entregarlo?
¿Cómo preparo este residuo para reciclar?
¿Tengo que secar el plástico?
¿Retiro los restos de comida del recipiente?
¿El papel debe estar limpio y seco?
Dime cómo limpiar la botella
¿Qué condiciones debe tener el material para entregarlo?
¿Hace falta vaciar el envase?
¿Debo quitar el líquido antes de separar?
Explica la preparación del cartón
¿Puedo entregar el recipiente con comida adentro?
¿Tengo que retirar los restos del metal?
¿Cómo dejo listo el material aprovechable?
¿Qué hago antes de depositar la botella?
¿Es necesario limpiar los residuos detectados?
¿Debo separar el contenido del envase?
Recomienda la preparación previa a la recolección
¿Qué debo hacer para que el objeto esté seco?
¿Cómo entrego los envases limpios?
¿Hay que quitar los residuos de comida?
¿Cómo acondiciono el vidrio para entregarlo?
¿Debo vaciar y secar este recipiente?
No pregunto dónde va, sino cómo lo limpio
Quiero preparar los materiales antes de reciclar""",
"val": """¿Qué limpieza necesita el envase antes de entregarlo?
Indica cómo acondicionar el material para la recolección
¿Hay que vaciar la botella para reciclarla?
¿Debo dejar seco este objeto?
Necesito preparar el recipiente que detectaste
¿Puedo entregar el cartón con restos de comida?""",
"test": """¿Qué pasos previos necesita el envase para su entrega?
Enséñame a dejar el residuo limpio y seco
¿Retiro la comida antes de llevarlo a reciclar?
¿En qué estado debo entregar el material aprovechable?
Quiero saber si debo limpiar la lata
¿Cómo elimino los restos del recipiente detectado?"""},
"incertidumbre": {
"train": """¿Qué tan seguro es el resultado?
¿El detector puede equivocarse?
Explica la confianza de la predicción
¿Por qué no estás seguro del material?
¿Qué limitaciones tiene esta clasificación?
¿La puntuación garantiza que sea correcto?
¿Puedes saber si está sucio por la foto?
¿Cómo verifico si la detección es fiable?
¿Qué hago si la etiqueta parece equivocada?
¿La cámara conoce la composición interna?
¿Por qué hay dudas sobre este residuo?
Aclara qué información no puede saber el sistema
¿Una confianza alta significa que nunca falla?
¿Puede confundirse plástico con vidrio?
¿Debo comprobar el objeto si tengo dudas?
¿Qué significa una confianza baja?
¿Se puede confiar en esta etiqueta?
¿Cómo afecta la iluminación a la predicción?
¿La imagen permite conocer la contaminación del residuo?
Necesito entender el margen de error
¿Por qué el modelo no reconoce bien este objeto?
¿El resultado es una certeza o una estimación?
No entiendo la incertidumbre del detector
¿Qué no se puede concluir de la fotografía?""",
"val": """¿Hay posibilidad de error en esa etiqueta?
Explica si el nivel de confianza es una garantía
¿Qué debo revisar cuando la predicción genera dudas?
¿Se puede inferir la limpieza solo mirando la imagen?
¿Por qué podría fallar el reconocimiento?
Necesito saber qué tan fiable es lo detectado""",
"test": """¿Puedo estar completamente seguro de esta clasificación?
¿La puntuación del modelo asegura el material real?
Cuéntame las limitaciones de lo que predijo la cámara
¿Qué datos no demuestra esta foto?
¿Es posible que el sistema haya confundido el residuo?
Ayúdame a interpretar la duda de la predicción"""},
"fuera_de_alcance": {
"train": """¿Qué tiempo hará mañana?
Escribe una canción de amor
¿Cuál es la capital de Francia?
Resuelve esta ecuación matemática
Recomiéndame una película
¿Qué puedo cocinar para la cena?
Cuéntame un chiste
¿Quién ganó el partido de fútbol?
Necesito un consejo para comprar acciones
Hazme un diagnóstico médico
Ayúdame a reservar un hotel
Programa un videojuego
¿Cuánto cuesta un celular?
Traduce una carta al inglés
¿Qué hora es en Japón?
Escribe un ensayo de historia
¿Cómo arreglo mi impresora?
Quiero organizar unas vacaciones
¿Dónde queda el restaurante más cercano?
Busca una noticia de política
¿Cómo cambio la contraseña del correo?
Redacta una invitación de cumpleaños
No hablo de basura, quiero aprender guitarra
¿Cuál es la raíz cuadrada de dieciséis?""",
"val": """¿Me puedes recomendar una serie de televisión?
Necesito ayuda para una tarea de geometría
Escribe un poema sobre la luna
¿Qué ingredientes lleva una sopa?
Quiero saber el precio de un computador
Consulta los resultados del torneo""",
"test": """¿Cómo consigo pasajes para viajar a otra ciudad?
Ayúdame a elegir una receta para el desayuno
¿Qué libro de aventuras recomiendas?
Explícame cómo instalar un juego
¿Qué temperatura hay hoy en Bogotá?
Escribe un mensaje para felicitar a mi amiga"""},
}


def main():
    carpeta = RAIZ / "datos" / "asistente"
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    vistas = set()
    cantidades = {}
    for split in ["train", "val", "test"]:
        registros = []
        for intencion, particiones in PREGUNTAS.items():
            for i, pregunta in enumerate(particiones[split].splitlines()):
                clave = " ".join(palabras(pregunta))
                if clave in vistas:
                    raise ValueError(f"Pregunta duplicada: {pregunta}")
                vistas.add(clave)
                contexto = [{"material": m} for m in rng.sample(MATERIALES, rng.randrange(4))]
                grupo = f"{intencion}:{split}:{i:03d}"
                # Aumentos únicamente de entrenamiento, agrupados con el original.
                variantes = [pregunta] if split != "train" else [pregunta, "Por favor, " + pregunta, "Quisiera consultar: " + pregunta]
                for variante in variantes:
                    registros.append({"id": f"{grupo}:{len(registros)}", "grupo": grupo,
                        "pregunta": variante, "intencion": intencion, "detecciones": contexto,
                        "contexto_texto": contexto_texto(contexto), "origen": "corpus_inicial_redactado_con_asistencia_IA",
                        "contexto_sintetico": True})
        (carpeta / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
        cantidades[split] = len(registros)
    reporte = {"semilla": 42, "cantidades": cantidades, "grupos_train": 144,
        "duplicados_normalizados_entre_preguntas_base": 0,
        "limitacion": "Corpus pequeño redactado con asistencia IA; prueba no recolectada con usuarios independientes. Requiere ampliación y evaluación externa antes de afirmar generalización.",
        "revision_humana": "pendiente", "contextos": "sintéticos, no detecciones reales"}
    (carpeta / "procedencia.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
