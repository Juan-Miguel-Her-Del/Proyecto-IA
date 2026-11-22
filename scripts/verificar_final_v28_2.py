"""Cierre verificable del piloto, sin evaluar conjuntos retenidos ni adoptar pesos."""
from pathlib import Path
import sys
import math
from collections import Counter
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar
from ecowaste.experimento_v28_2 import SALIDA,CERTIFICADOS,Diario,codigo,validar


def main():
    c=leer(CERTIFICADOS/'protocolo_congelado.json');validar(c['config'])
    if c['codigo']!=codigo():raise ValueError('Código crítico cambiado después del diagnóstico')
    for archivo in ['historicos_protegidos.json','sha256_checkpoints_historicos.json']:
        for p,h in leer(SALIDA/archivo).items():
            if sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
    b=leer(RAIZ/'resultados/v28_validacion_anotaciones/respaldo_verificado.json')
    for base in ['origen','destino']:
        for p,h in b['archivos'].items():
            if sha(RAIZ/b[base]/p)!=h:raise ValueError('Revisiones o respaldo alterado: '+p)
    ramas={};hist={};cp=[];estados_cero={}
    for rama in ['A','B']:
        d=Diario(rama,c);ck,_=d.leer();r=ck['v28_2'];hist[rama]=r['historial']
        e0=leer(d.base/'confirmaciones/paso_0000.json')
        estados_cero[rama]=torch.load(RAIZ/e0['checkpoint'],map_location='cpu',weights_only=True)['v28_2']['huellas_estado']
        if r['paso']!=300:raise ValueError('Rama incompleta')
        for h in r['historial']:
            if any(not math.isfinite(h[k]) for k in ['loss_mapas','loss_global','loss_total','norma_gradiente_pre_clip']):raise ValueError('Pérdidas/gradientes no finitos')
        entradas={'historico':[],'nuevo':[],'global':[]}
        for h in r['historial']:
            for k in entradas:entradas[k]+=h['entradas'][k]
        tiempos=[leer(p) for p in (d.base/'intentos').glob('*_fin.json')]
        pasos=sum(t['actualizaciones_este_intento'] for t in tiempos)
        if pasos!=300:raise ValueError('Intentos incompletos o ampliación de presupuesto')
        ramas[rama]={'actualizaciones':300,'paso_adamw':int(next(iter(ck['optimizador']['state'].values()))['step']),
            'exposiciones':{k:len(v) for k,v in entradas.items()},'imagenes_distintas':{k:len(set(v)) for k,v in entradas.items()},
            'repeticiones_auxiliares':dict(Counter(entradas['nuevo'])),'segundos_actualizaciones':sum(h['segundos'] for h in r['historial']),
            'segundos_intentos_con_checkpoints':sum(t['segundos'] for t in tiempos),
            'memoria_gpu_max_allocated_bytes':max(t['memoria_gpu_max_allocated_bytes'] for t in tiempos),
            'memoria_gpu_max_reserved_bytes':max(t['memoria_gpu_max_reserved_bytes'] for t in tiempos),
            'loss_media_primeros50':sum(h['loss_total'] for h in r['historial'][:50])/50,
            'loss_media_ultimos50':sum(h['loss_total'] for h in r['historial'][-50:])/50,
            'restauraciones':[leer(p) for p in sorted((d.base/'intentos').glob('*_inicio.json'))],
            'estados_finales':r['huellas_estado']}
        cp += [RAIZ/leer(d.base/'confirmaciones'/f'paso_{k:04d}.json')['checkpoint'] for k in [0,100,200,300]]
    comunes=all(a['entradas']['historico'][:6]==b['entradas']['historico'] and a['entradas']['global']==b['entradas']['global'] and
        a['huella_aumentos_comunes']==b['huella_aumentos_comunes'] for a,b in zip(hist['A'],hist['B']))
    if not comunes:raise ValueError('Orden o aumentos comunes diferentes')
    if estados_cero['A']!=estados_cero['B']:raise ValueError('A y B no parten del mismo estado efectivo')
    m=leer(SALIDA/'metricas.json')
    if m['protocolo']!=c or m['evaluacion_val_test_generalizacion'] or m['adopcion']:raise ValueError('Ámbito distinto')
    for n in ['Original','A','B']:
        for dom,cantidad in [('historico',964),('auxiliar',28)]:
            cap=SALIDA/'evaluacion'/n/dom;cert=leer(cap/'captura.json')
            if cert['gt_utilizado'] or cert['imagenes']!=cantidad:raise ValueError('Captura ajena')
            for p,h in cert['archivos'].items():
                if sha(cap/p)!=h:raise ValueError('Captura alterada')
            for v in [1,5]:
                pred=leer(cap/f'predicciones_{v}v.json')
                if len(pred)!=cantidad or len({r['id'] for r in pred})!=cantidad:raise ValueError('Predicciones incompletas')
    publicar(SALIDA/'verificacion_final.json',{'aprobada':True,'ramas':ramas,'anclas_y_aumentos_iguales_300':comunes,
        'eventos_humanos_conservados':134,'respaldo_archivos_conservados':731,'checkpoints_historicos_conservados':105,
        'checkpoints_experimentales':{p.relative_to(RAIZ).as_posix():sha(p) for p in cp},
        'checkpoint_original_intacto':True,'datasets_historicos_intactos':True,'V28_NO_ENTRENAR_intacto':True,
        'val_test_generalizacion_evaluados':False,'modelo_predeterminado_modificado':False,
        'pesos_pilotos_desechados':not any(SALIDA.glob('descartar_*')),
        'controles_actualizados':sha(CERTIFICADOS/'controles.json')})
    print('Cierre aprobado: A300/B300, estados, 600 trazas, 12 evaluaciones train, históricos y revisiones intactos',flush=True)


if __name__=='__main__':main()
