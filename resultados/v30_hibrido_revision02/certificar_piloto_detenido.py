"""Cierre adicional del piloto: capturas, métricas, presupuesto e inicio propio."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import math
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v30 import SALIDA, CONFIG, codigo, Diario
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.ablacion_fondo import huella


def main():
    c=leer(SALIDA/'protocolo.json');gate=leer(SALIDA/'piloto_800/puerta.json');met=leer(SALIDA/'metricas_piloto.json')
    if c['codigo']!=codigo() or c['config']!=leer(CONFIG) or gate['aprobada'] or gate['contrato']!=c:
        raise ValueError('Código, protocolo o decisión distintos')
    if met['paso_evaluado']!=800 or met['presupuesto_completo_ejecutado'] or met['adopcion']:
        raise ValueError('No es el piloto detenido')
    ramas={}
    plan=torch.load(SALIDA/'plan.pt',weights_only=True)
    for n in ['A','B']:
        ck,_=Diario(n,c).leer();ramas[n]=ck
        if ck['paso']!=800:raise ValueError('Presupuesto alterado')
        if huella(plan)!=c['plan']:raise ValueError('Plan distinto')
        for k,h in enumerate(ck['historial']):
            if any(h['indices'][nombre]!=p[k].tolist() for nombre,p in plan.items()):raise ValueError('Entradas fuera de plan')
        if any((RAIZ/f'checkpoints/v30_hibrido_revision02/{n}/paso_{k:04d}.pt').exists() for k in [1600,2400]):
            raise ValueError('Se continuó tras fallo de puerta')
    if any(a['entradas_sha256']!=b['entradas_sha256'] for a,b in zip(ramas['A']['historial'],ramas['B']['historial'])):
        raise ValueError('Entradas A/B diferentes')
    originales=leer(RAIZ/'resultados/v29_packwise_click/metricas.json')
    if originales['backend']!=met['backend']:raise ValueError('Backend distinto en referencia reutilizada')
    archivos={};cohortes={'historico_train':2505,'nuevo_train':4624,'desarrollo':361,'externo_retenido':1090}
    for n in ['Original','A','B']:
        for dom,gt in cohortes.items():
            r=met['resultados'][n][dom];q=RAIZ/r['analisis'];folder=RAIZ/r['captura']
            if sha(q)!=r['sha256_analisis']:raise ValueError('Análisis alterado')
            cert=leer(folder/'captura.json')
            for p,h in cert['archivos'].items():
                if sha(folder/p)!=h:raise ValueError('Captura alterada')
                archivos[(folder/p).relative_to(RAIZ).as_posix()]=h
            archivos[q.relative_to(RAIZ).as_posix()]=sha(q)
            for v in ['1','5']:
                m=r['metricas'][v];aps=[a for a in m['ap50_por_material'].values() if a is not None]
                if m['tp50']+m['fn50']!=gt or abs(sum(aps)/len(aps)-m['map50'])>1e-12:
                    raise ValueError('Conteos/mAP inconsistentes')
                if any(not math.isfinite(m[k]) or not 0<=m[k]<=1 for k in ['map50','map50_95','recall50','precision50']):
                    raise ValueError('Métrica inválida')
    reproduccion={}
    for n in ['A','B']:
        anterior=leer(SALIDA/f'piloto_800/{n}/predicciones.json')
        posterior=leer(RAIZ/met['resultados'][n]['desarrollo']['captura']/'predicciones_1v.json')
        exacto=anterior==[{k:v for k,v in p.items() if k!='id'} for p in posterior]
        reproduccion[n]={'predicciones_desarrollo_1v_exactas':exacto}
        if not exacto:raise ValueError('La evaluación del piloto no se reprodujo exactamente')
    inicial=leer(SALIDA/'certificado_inicializacion_desde_cero.json')
    if not all(r['pesos_recreados_exactos'] and r['adamw_vacio'] for r in inicial.values()):raise ValueError('Inicialización no comprobada')
    for r in leer(SALIDA/'visuales.json'):
        if sha(RAIZ/r['archivo'])!=r['sha256']:raise ValueError('Visual alterado')
        archivos[r['archivo']]=r['sha256']
    publicar(SALIDA/'certificado_piloto_detenido.json',{'aprobada':True,'presupuesto_maximo':2400,'pasos_reales_por_rama':800,
        'avance_tras_fallo':False,'entradas_A_B_iguales':True,'inicializacion':inicial,'reproduccion_piloto':reproduccion,
        'capturas_y_analisis_y_visuales':archivos,'test_evaluado':False,'generalizacion_desbloqueada':False,'adopcion':False})
    print('Certificado posthoc aprobado',reproduccion,flush=True)


if __name__=='__main__':main()
