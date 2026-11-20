"""Auditoría CPU de cajas guardadas: cobertura, geometría, material y tamaño de GT."""
import torch
from ecowaste.cajas import area,intersecciones,iou_pares,cubiertas_por_ignorar

ESTADOS=['correcto','sin_cobertura_suficiente','geometria_incorrecta','material_equivocado']
TAMANOS=['menos_8_px','8_a_16_px','16_a_32_px','32_px_o_mas']


def tamano_objeto(box,lado=384):
    w,h=((box[2:]-box[:2])*lado).tolist()
    menor=min(w,h)
    grupo=TAMANOS[0] if menor<8 else TAMANOS[1] if menor<16 else TAMANOS[2] if menor<32 else TAMANOS[3]
    return {'ancho_384':w,'alto_384':h,'lado_menor_384':menor,'area_384':w*h,'grupo_tamano':grupo}


def auditar_imagen(pred,objetivo,umbral=.05,cobertura_suficiente=.5):
    """Usa la asignación del evaluador; la región observable es su caja final."""
    if not 0<=umbral<=1 or not 0<cobertura_suficiente<=1:raise ValueError('Umbral de auditoría inválido')
    boxes=torch.as_tensor(pred['boxes'],dtype=torch.float32,device='cpu').reshape(-1,4)
    scores=torch.as_tensor(pred['scores'],dtype=torch.float32,device='cpu')
    labels=torch.as_tensor(pred['labels'],dtype=torch.long,device='cpu')
    if not len(boxes)==len(scores)==len(labels):raise ValueError('Predicciones incompletas')
    gt=objetivo['boxes'].cpu().float();reales=objetivo['labels'].cpu().long()
    ix=torch.where(scores>=umbral)[0]
    b,s,l=boxes[ix],scores[ix],labels[ix]
    ious=iou_pares(b,gt);inter=intersecciones(b,gt)
    cobertura=inter/area(gt)[None].clamp(min=1e-9)
    omitir=cubiertas_por_ignorar(b,objetivo['ignore_boxes'].cpu().float())
    max_iou=ious.max(1).values if len(gt) else torch.zeros(len(b))
    ignorables=omitir&(max_iou<.5)
    usados=set();parejas={};fp=[];ignoradas=[]
    orden=sorted(range(len(b)),key=lambda j:-float(s[j]))
    for j in orden:
        libres=[k for k in range(len(gt)) if int(reales[k])==int(l[j]) and k not in usados and float(ious[j,k])>=.5]
        if libres:
            k=max(libres,key=lambda k:float(ious[j,k]));usados.add(k);parejas[k]=j
        elif bool(ignorables[j]):ignoradas.append(j)
        else:fp.append(j)
    elegibles=[j for j in range(len(b)) if not bool(ignorables[j])]
    objetos=[]
    for k,g in enumerate(gt):
        razonables=[j for j in elegibles if float(ious[j,k])>=.5]
        correctos=[j for j in razonables if int(l[j])==int(reales[k])]
        cubren=[j for j in elegibles if float(cobertura[j,k])>=cobertura_suficiente]
        maxj=max(elegibles,key=lambda j:(float(ious[j,k]),float(s[j]),-j)) if elegibles else None
        maxcorrecto=max((float(ious[j,k]) for j in elegibles if int(l[j])==int(reales[k])),default=0.)
        maxcov=max((float(cobertura[j,k]) for j in elegibles),default=0.)
        subtipo=None
        if k in parejas:estado='correcto';testigo=parejas[k]
        elif correctos:
            estado='geometria_incorrecta';subtipo='caja_compartida_sin_asignacion_univoca'
            testigo=max(correctos,key=lambda j:float(ious[j,k]))
        elif razonables:
            estado='material_equivocado';testigo=max(razonables,key=lambda j:(float(ious[j,k]),float(s[j]),-j))
        elif cubren:
            estado='geometria_incorrecta';testigo=max(cubren,key=lambda j:(float(ious[j,k]),float(s[j]),-j))
            ratio=float(area(b[testigo:testigo+1])[0]/area(g[None])[0])
            subtipo='sobredimensionada' if ratio>2 else 'subdimensionada' if ratio<.75 else 'forma_o_desplazamiento'
        else:estado='sin_cobertura_suficiente';testigo=maxj
        j=testigo
        fila={'indice_gt':k,'clase_real':int(reales[k]),'estado':estado,'subtipo':subtipo,
            **tamano_objeto(g),'iou_max_cualquier_material':float(ious[maxj,k]) if maxj is not None else 0.,
            'iou_max_material_correcto':maxcorrecto,'cobertura_max_gt':maxcov,
            'indice_prediccion_testigo':int(ix[j]) if j is not None else None,
            'material_testigo':int(l[j]) if j is not None else None,
            'puntuacion_testigo':float(s[j]) if j is not None else None,
            'iou_testigo':float(ious[j,k]) if j is not None else None,
            'cobertura_gt_testigo':float(cobertura[j,k]) if j is not None else None,
            'ratio_area_pred_gt':float(area(b[j:j+1])[0]/area(g[None])[0]) if j is not None else None,
            'caja_gt':g.tolist(),'caja_testigo':b[j].tolist() if j is not None else None,
            'cajas_razonables_cualquier_material':len(razonables),'cajas_razonables_material_correcto':len(correctos)}
        objetos.append(fila)
    falsos=[]
    for j in fp:
        dup=[k for k in usados if int(reales[k])==int(l[j]) and float(ious[j,k])>=.5]
        mal=[k for k in range(len(gt)) if int(reales[k])!=int(l[j]) and float(ious[j,k])>=.5]
        cubierta=[k for k in range(len(gt)) if float(cobertura[j,k])>=cobertura_suficiente]
        tipo='duplicado' if dup else 'material_equivocado' if mal else 'geometria_incorrecta' if cubierta else 'fondo_o_solape_debil'
        falsos.append({'indice_prediccion':int(ix[j]),'clase_predicha':int(l[j]),'tipo_fp':tipo,
                      'puntuacion':float(s[j]),'caja':b[j].tolist()})
    return {'objetos':objetos,'falsos_positivos':falsos,'tp':len(parejas),'fp':len(fp),
            'fn':len(gt)-len(parejas),'ignoradas':len(ignoradas),'cajas':len(b)}
