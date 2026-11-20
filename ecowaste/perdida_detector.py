"""Asignación de centro/escala y pérdida equilibrada con regiones ignoradas."""
import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.cajas import cubiertas_por_ignorar, giou_alineado
from ecowaste.detector import decodificar


def asignar(objetivo, lado=48, candidatos=2, umbral_escala=0.1, radio_centros=0):
    boxes = objetivo["boxes"].cpu()
    tamanos = boxes[:, 2:] - boxes[:, :2]
    escalas = tamanos.prod(-1).sqrt()
    ocupados, indices, objetos = set(), [], []
    orden = escalas.argsort(stable=True).tolist()
    celdas = {}
    def colocar(i, y, x):
        preferido = 0 if float(escalas[i]) <= umbral_escala else candidatos - 1
        opciones = [preferido] + [k for k in range(candidatos) if k != preferido]
        for k in opciones:
            indice = (y * lado + x) * candidatos + k
            if indice not in ocupados:
                ocupados.add(indice)
                indices.append(indice)
                objetos.append(i)
                return True
        return False
    # Reservar primero todos los centros evita que los vecinos de un objeto desplacen otro.
    for i in orden:
        centro = (boxes[i, :2] + boxes[i, 2:]) / 2
        x, y = (centro * lado).long().clamp(0, lado - 1).tolist()
        celdas[i] = (y, x)
        colocar(i, y, x)
    if radio_centros:
        for i in orden:
            cy, cx = celdas[i]
            vecinos = [(dy, dx) for dy in range(-radio_centros, radio_centros + 1) for dx in range(-radio_centros, radio_centros + 1) if dy or dx]
            vecinos.sort(key=lambda t: t[0] ** 2 + t[1] ** 2)
            for dy, dx in vecinos:
                y, x = cy + dy, cx + dx
                if not (0 <= y < lado and 0 <= x < lado):
                    continue
                punto = boxes.new_tensor([(x + .5) / lado, (y + .5) / lado])
                if bool(((punto >= boxes[i, :2]) & (punto <= boxes[i, 2:])).all()):
                    colocar(i, y, x)
    colisiones = [i for i in orden if i not in set(objetos)]
    return {"indices": torch.tensor(indices, dtype=torch.long), "objetos": torch.tensor(objetos, dtype=torch.long), "colisiones": colisiones}


def priores_train(objetivos):
    tamanos = torch.cat([o["boxes"][:, 2:] - o["boxes"][:, :2] for o in objetivos])
    escalas = tamanos.prod(-1).sqrt()
    corte = float(escalas.median())
    priors = []
    for mascara in [escalas <= corte, escalas > corte]:
        valores = tamanos[mascara] if bool(mascara.any()) else tamanos
        priors.append(valores.median(dim=0).values.clamp(1e-3, 0.99).tolist())
    return priors, corte


def cajas_centro(cajas):
    return torch.cat([(cajas[..., :2] + cajas[..., 2:]) / 2, cajas[..., 2:] - cajas[..., :2]], -1)


class PerdidaDetector(nn.Module):
    def __init__(self, umbral_escala=0.1, pesos_clase=None, ignorar_interiores_gt=False, radio_centros=0, normalizacion_clases="por_imagen"):
        super().__init__()
        self.umbral_escala = umbral_escala
        self.ignorar_interiores_gt = ignorar_interiores_gt
        self.radio_centros = radio_centros
        if normalizacion_clases not in ["por_imagen", "global"]:
            raise ValueError("Normalización de clases no reconocida")
        self.normalizacion_clases = normalizacion_clases
        self.register_buffer("pesos_clase", pesos_clase)

    def forward(self, salida, objetivos):
        cajas, presencia, clases = decodificar(salida, self.radio_centros)
        terminos = {k: salida.float().sum() * 0 for k in ["obj", "cls", "box", "giou"]}
        clase_numerador, clase_denominador = salida.float().sum() * 0, salida.new_zeros((), dtype=torch.float32)
        positivos, negativos, colisiones, ignorados = 0, 0, 0, 0
        for b, objetivo in enumerate(objetivos):
            box = cajas[b].reshape(-1, 4)
            obj = presencia[b].reshape(-1)
            cls = clases[b].reshape(-1, clases.shape[-1])
            asignacion = objetivo.get("asignacion")
            if asignacion is None:
                asignacion = asignar(objetivo, salida.shape[1], salida.shape[3], self.umbral_escala, self.radio_centros)
            ix = asignacion["indices"].to(salida.device)
            gt = objetivo["boxes"].to(salida.device)
            etiquetas = objetivo["labels"].to(salida.device)
            usados = asignacion["objetos"].to(salida.device)
            regiones = objetivo["ignore_boxes"].to(salida.device)
            if asignacion["colisiones"]:
                regiones = torch.cat([regiones, gt[asignacion["colisiones"]]])
            with torch.no_grad():
                omitir = cubiertas_por_ignorar(box.detach(), regiones)
                if self.ignorar_interiores_gt and len(gt):
                    # Solo centros dentro de la anotación: no cambia las cajas ni crea positivos nuevos.
                    centros = (box.detach()[:, :2] + box.detach()[:, 2:]) / 2
                    omitir |= ((centros[:, None] >= gt[None, :, :2]) & (centros[:, None] <= gt[None, :, 2:])).all(-1).any(-1)
                g = objetivo.get("geometria")
                if g:
                    centro = (box[:, :2] + box[:, 2:]) / 2
                    inicio = box.new_tensor([g["relleno_x"], g["relleno_y"]]) / g["lado"]
                    fin = inicio + box.new_tensor([g["ancho_original"] * g["escala_x"], g["alto_original"] * g["escala_y"]]) / g["lado"]
                    omitir |= ((centro < inicio) | (centro > fin)).any(-1)
                mascara_neg = ~omitir
                mascara_neg[ix] = False
                candidatos_neg = mascara_neg.nonzero().flatten()
                cantidad_neg = min(len(candidatos_neg), max(32, len(usados.unique()) * 10))
                # Mezcla de negativos difíciles y aleatorios, sin reemplazo.
                n_dificiles = cantidad_neg // 2
                orden = obj[candidatos_neg].argsort(descending=True)
                dificiles = candidatos_neg[orden[:n_dificiles]]
                resto = candidatos_neg[orden[n_dificiles:]]
                azar = resto[torch.randperm(len(resto), device=salida.device)[:cantidad_neg - n_dificiles]]
                indices_neg = torch.cat([dificiles, azar])
            if len(ix):
                pesos_objeto = torch.bincount(usados, minlength=len(gt))[usados].float().reciprocal()
                if self.normalizacion_clases == "global":
                    ce_global = F.cross_entropy(cls[ix], etiquetas[usados], weight=self.pesos_clase, reduction="none")
                    clase_numerador = clase_numerador + (ce_global * pesos_objeto).sum()
                    clase_denominador = clase_denominador + (pesos_objeto.sum() if self.pesos_clase is None else (pesos_objeto * self.pesos_clase[etiquetas[usados]]).sum())
                if self.radio_centros == 0:
                    terminos["obj"] = terminos["obj"] + F.binary_cross_entropy_with_logits(obj[ix], torch.ones_like(obj[ix]))
                    if self.normalizacion_clases == "por_imagen":
                        terminos["cls"] = terminos["cls"] + F.cross_entropy(cls[ix], etiquetas[usados], weight=self.pesos_clase)
                    terminos["box"] = terminos["box"] + F.smooth_l1_loss(cajas_centro(box[ix]), cajas_centro(gt[usados]), beta=0.05)
                    terminos["giou"] = terminos["giou"] + (1 - giou_alineado(box[ix], gt[usados])).mean()
                else:
                    # Cada objeto conserva peso total uno aunque supervise varias celdas.
                    def media_objeto(valores):
                        return (valores * pesos_objeto).sum() / pesos_objeto.sum()
                    terminos["obj"] = terminos["obj"] + media_objeto(F.binary_cross_entropy_with_logits(obj[ix], torch.ones_like(obj[ix]), reduction="none"))
                    ce = F.cross_entropy(cls[ix], etiquetas[usados], weight=self.pesos_clase, reduction="none")
                    denominador = pesos_objeto.sum() if self.pesos_clase is None else (pesos_objeto * self.pesos_clase[etiquetas[usados]]).sum()
                    if self.normalizacion_clases == "por_imagen":
                        terminos["cls"] = terminos["cls"] + (ce * pesos_objeto).sum() / denominador
                    terminos["box"] = terminos["box"] + media_objeto(F.smooth_l1_loss(cajas_centro(box[ix]), cajas_centro(gt[usados]), beta=0.05, reduction="none").mean(-1))
                    terminos["giou"] = terminos["giou"] + media_objeto(1 - giou_alineado(box[ix], gt[usados]))
            if len(indices_neg):
                terminos["obj"] = terminos["obj"] + F.binary_cross_entropy_with_logits(obj[indices_neg], torch.zeros_like(obj[indices_neg]))
            positivos += len(ix)
            negativos += len(indices_neg)
            ignorados += int(omitir.sum().item())
            colisiones += len(asignacion["colisiones"])
        terminos = {k: v / len(objetivos) for k, v in terminos.items()}
        if self.normalizacion_clases == "global" and positivos:
            terminos["cls"] = clase_numerador / clase_denominador.clamp(min=1e-9)
        total = terminos["obj"] + terminos["cls"] + 5 * terminos["box"] + 2 * terminos["giou"]
        detalle = {k: float(v.detach()) for k, v in terminos.items()}
        detalle.update(positivos=positivos, negativos=negativos, colisiones=colisiones, ignorados=ignorados)
        return total, detalle
