"""Preservación de clasificador/detector históricos y cierre V31."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.experimento_v31 import OUT,CONFIG,codigo,Diario
from ecowaste.incorporacion import sha,leer
from ecowaste.reanudacion_regional import publicar


def main():
    c=leer(OUT/'protocolo.json')
    if c['codigo']!=codigo() or c['cfg']!=leer(CONFIG):raise ValueError('Código/configuración distintos')
    protegidos=leer(OUT/'protegidos.json')
    for k,(p,h) in enumerate(protegidos.items()):
        if sha(RAIZ/p)!=h:raise ValueError('Histórico modificado: '+p)
        if (k+1)%2000==0:print('Protegidos',k+1,'/',len(protegidos),flush=True)
    for p,h in c['datos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Datos alterados')
    pr=leer(OUT/'pruebas.json');co=leer(OUT/'controles/reanudacion.json');me=leer(OUT/'controles/memorizacion.json')
    if not pr['aprobada'] or not co['aprobada']:raise ValueError('Controles fallidos')
    pasos={}
    for r in ['A','B']:
        ck,_=Diario(r,c).leer();pasos[r]=ck['paso'] if ck else 0
    if me['aprobada'] and any(k!=2400 for k in pasos.values()):raise ValueError('Presupuesto incompleto')
    if me['aprobada']:
        a,_=Diario('A',c).leer();b,_=Diario('B',c).leer()
        if any(x['entradas_sha256']!=y['entradas_sha256'] for x,y in zip(a['historial'],b['historial'])):raise ValueError('Entradas A/B distintas')
        for r in ['A','B']:
            for k in [800,1600,2400]:
                e=leer(OUT/f'evaluacion/{r}/{k:04d}/resumen.json')
                for d in e['resultados'].values():
                    q=RAIZ/d['analisis']
                    if sha(q)!=d['sha256']:raise ValueError('Análisis alterado')
                    cap=leer(q.parent/'captura.json')
                    for p,h in cap['archivos'].items():
                        if sha(q.parent/p)!=h:raise ValueError('Predicción alterada')
    archivos={**codigo(),CONFIG.relative_to(RAIZ).as_posix():sha(CONFIG)}
    for p in OUT.rglob('*'):
        if p.is_file() and p.name not in ['integridad.json','recibo_integridad.json'] and p.suffix not in ['.lock','.pyc']:archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    for p in (RAIZ/'checkpoints/v31_instancias').rglob('*.pt'):archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    doc=RAIZ/'documentacion/31_deteccion_por_instancias.md';archivos[doc.relative_to(RAIZ).as_posix()]=sha(doc)
    out={'aprobada':True,'preservados':len(protegidos),'checkpoints_previos':sum(p.startswith('checkpoints/') for p in protegidos),
         'pasos':pasos,'memorizacion_aprobada':me['aprobada'],'pruebas':pr['resultados'],'test_evaluado':False,'adopcion_automatica':False,
         'clasificador_modificado':False,'predeterminado_modificado':False,'archivos':archivos}
    publicar(OUT/'integridad.json',out);publicar(OUT/'recibo_integridad.json',{'sha256':sha(OUT/'integridad.json')});print({k:v for k,v in out.items() if k!='archivos'},flush=True)


if __name__=='__main__':main()
