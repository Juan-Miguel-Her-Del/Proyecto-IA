"""Transformer de intenciones y respuestas apoyadas en detecciones estructuradas."""
import math
from pathlib import Path
import re
import unicodedata

import torch
from torch import nn
from ecowaste.transformer import Atencion, Bloque

INTENCIONES = ["resumen", "material", "separacion", "preparacion", "incertidumbre", "fuera_de_alcance"]
MATERIALES = ["plastico", "papel", "carton", "vidrio", "metal", "organico"]
FUENTE_REGLAS = "https://www.minambiente.gov.co/preguntas-frecuentes/"


def palabras(texto):
    normal = unicodedata.normalize("NFKD", texto.casefold())
    normal = "".join(c for c in normal if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9_]+", normal)


class Tokenizador:
    def __init__(self, vocabulario):
        self.vocabulario = vocabulario

    @classmethod
    def ajustar(cls, registros):
        # Solo recibe la partición de entrenamiento.
        especiales = ["PAD", "UNK", "SEP"]
        vocabulario = {w: i for i, w in enumerate(especiales)}
        for palabra in sorted({p for r in registros for p in palabras(r["pregunta"]) + palabras(r["contexto_texto"])}):
            vocabulario.setdefault(palabra, len(vocabulario))
        return cls(vocabulario)

    def codificar(self, pregunta, contexto, longitud=64):
        # Reserva contexto aunque la pregunta sea larga.
        secuencia = palabras(pregunta)[:longitud - 17] + ["SEP"] + palabras(contexto)[:16]
        ids = [self.vocabulario.get(p, 1) for p in secuencia]
        return ids + [0] * (longitud - len(ids))


class TransformerIntenciones(nn.Module):
    def __init__(self, vocabulario, dimension=64, bloques=2, cabezas=4, mlp=128, longitud=64):
        super().__init__()
        self.configuracion = dict(vocabulario=vocabulario, dimension=dimension, bloques=bloques, cabezas=cabezas, mlp=mlp, longitud=longitud)
        self.embedding = nn.Embedding(vocabulario, dimension, padding_idx=0)
        posicion = torch.arange(longitud).unsqueeze(1)
        frecuencia = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        codificacion = torch.zeros(longitud, dimension)
        codificacion[:, 0::2] = torch.sin(posicion * frecuencia)
        codificacion[:, 1::2] = torch.cos(posicion * frecuencia)
        self.register_buffer("posicion", codificacion)
        self.bloques = nn.ModuleList([Bloque(dimension, cabezas, mlp) for _ in range(bloques)])
        self.norma = nn.LayerNorm(dimension)
        self.cabezal = nn.Linear(dimension, len(INTENCIONES))

    def forward(self, ids):
        validos = ids != 0
        if not bool(validos.any(dim=1).all()):
            raise ValueError("Cada entrada necesita al menos un token válido")
        x = self.embedding(ids) + self.posicion[:ids.shape[1]]
        for bloque in self.bloques:
            x = bloque(x, validos)
        x = self.norma(x)
        vector = (x * validos.unsqueeze(-1)).sum(1) / validos.sum(1, keepdim=True)
        return self.cabezal(vector)


def contexto_texto(detecciones):
    cantidades = {m: 0 for m in MATERIALES}
    for objeto in detecciones:
        if objeto.get("material") in cantidades:
            cantidades[objeto["material"]] += 1
    return "contexto " + " ".join(f"{m} {min(n, 9)}" for m, n in cantidades.items())


def responder(intencion, detecciones, pregunta=""):
    """Las reglas nunca agregan detecciones ni propiedades no observadas."""
    if intencion == "fuera_de_alcance":
        return "Puedo explicar las detecciones de residuos y orientar sobre su separación."
    objetos = [d for d in detecciones if d.get("material") in MATERIALES]
    referidos = set(palabras(pregunta)) & set(MATERIALES)
    if referidos:
        objetos = [d for d in objetos if d["material"] in referidos]
    if not objetos:
        return "No tengo detecciones de esos residuos para explicar. Necesito una imagen analizada o que selecciones un residuo detectado."
    conteo = {m: sum(d["material"] == m for d in objetos) for m in MATERIALES}
    resumen = ", ".join(f"{n} de {m}" for m, n in conteo.items() if n)
    if intencion == "resumen":
        return f"En los resultados suministrados hay {len(objetos)} detecciones: {resumen}. Es un conteo de esta imagen o fotograma."
    if intencion == "material":
        return f"Los materiales predichos son: {resumen}. Orgánico representa restos alimentarios o vegetales; las otras cinco clases se agrupan operativamente como inorgánicas."
    if intencion == "incertidumbre":
        return f"El modelo predice: {resumen}. La puntuación del modelo no garantiza que acierte. La fotografía no demuestra composición interna, limpieza ni aprovechabilidad; revisa el objeto si hay duda."
    partes = []
    for m in MATERIALES:
        seleccion = [d for d in objetos if d["material"] == m]
        if not seleccion:
            continue
        if any(d.get("peligroso") is True for d in seleccion):
            partes.append(f"{m}: se indicó un posible residuo peligroso; requiere orientación de un gestor especializado y no una recomendación de separación ordinaria.")
        elif m == "organico":
            partes.append("Orgánico: si son restos alimentarios o vegetales aprovechables, corresponde al verde según el sistema local de recolección o aprovechamiento.")
        elif intencion == "preparacion":
            partes.append(f"{m}: si el residuo es aprovechable, entrégalo sin restos y seco; verifica si tu servicio local recibe ese objeto. No se ha determinado su estado a partir de la imagen.")
        elif all(d.get("limpio") is True and d.get("aprovechable") is True for d in seleccion):
            partes.append(f"{m}: con la limpieza y aprovechabilidad que indicaste, corresponde al blanco.")
        else:
            partes.append(f"{m}: si es aprovechable corresponde al blanco; si es un residuo ordinario no aprovechable corresponde al negro. ¿Está limpio y lo recibe el sistema local de aprovechamiento?")
    return f"Predicciones: {resumen}. " + " ".join(partes) + f" Fuente del código de colores: {FUENTE_REGLAS}"


class Asistente:
    def __init__(self, checkpoint, dispositivo="cpu"):
        datos = torch.load(Path(checkpoint), map_location=dispositivo, weights_only=True)
        if datos.get("tipo") != "ecowaste_intenciones_desde_cero_v1":
            raise ValueError("Checkpoint no compatible")
        self.tokenizador = Tokenizador(datos["vocabulario"])
        self.modelo = TransformerIntenciones(**datos["configuracion"]).to(dispositivo).eval()
        self.modelo.load_state_dict(datos["pesos"])
        self.umbral = datos["umbral"]
        self.dispositivo = dispositivo

    @torch.inference_mode()
    def consultar(self, pregunta, detecciones):
        ids = torch.tensor([self.tokenizador.codificar(pregunta, contexto_texto(detecciones))], device=self.dispositivo)
        probabilidades = self.modelo(ids).softmax(-1)[0]
        confianza, indice = probabilidades.max(0)
        intencion = INTENCIONES[indice.item()]
        tokens = palabras(pregunta)
        proporcion_desconocida = sum(t not in self.tokenizador.vocabulario for t in tokens) / max(len(tokens), 1)
        if not tokens or confianza.item() < self.umbral or proporcion_desconocida > 0.5:
            return {"intencion": "abstencion", "puntuacion": confianza.item(), "respuesta": "No interpreté la pregunta con suficiente claridad. Puedes preguntar por el material, la separación o los resultados detectados."}
        return {"intencion": intencion, "puntuacion": confianza.item(), "respuesta": responder(intencion, detecciones, pregunta)}
