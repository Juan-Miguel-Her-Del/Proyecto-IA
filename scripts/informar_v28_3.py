"""Inventario de trabajo pendiente: potencial no acreditado, nunca nuevas anotaciones."""
from collections import Counter
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from scripts.auditar_v28_3 import gestion,SALIDA
from scripts.anotar_v27_2 import PROYECTO
from ecowaste.auxiliares_publicos import Taller
from ecowaste.correcciones_v28_3 import auditorias_verificadas
from ecowaste.incorporacion import leer,publicar,sha


def main():
    t=Taller(PROYECTO,RAIZ);g=gestion(t);ps=auditorias_verificadas(SALIDA/'auditorias');p=ps[-1]
    imgs=leer(p.parent/'imagenes.json');incs=g.listar();filas=[]
    for f in imgs:
        ins=[r for r in incs if r['imagen']==f['imagen']];validas=[o for o in f['qa']['objetos'] if not o['descartada'] and o['valido_geometria']]
        actuales=Counter(o['material'] for o in validas);regiones=Counter(o['material'] for o in validas if o['componente_maximo_48']>=2)
        if ins:filas.append({'imagen':f['imagen'],'prioridad':min(r['prioridad'] for r in ins),'candidata_actual':f['candidata_exploratoria'],
            'guardado':f['guardado'],'incidencias_bloqueantes':len([r for r in ins if r['bloqueante'] and r['estado']!='corregida']),
            'objetos_guardados_validos_no_liberados':dict(actuales),'regiones_48_potenciales_no_acreditadas':dict(regiones),
            'enlace':'http://127.0.0.1:8768/?imagen='+f['imagen'].replace(':','%3A')})
    filas.sort(key=lambda f:(f['prioridad'],f['incidencias_bloqueantes'],f['imagen']))
    recibo={'sha256_auditoria':sha(p),'auditoria':p.relative_to(RAIZ).as_posix(),'plan':filas,
        'opciones_metal':sorted([f for f in filas if not f['candidata_actual'] and f['regiones_48_potenciales_no_acreditadas'].get('metal',0)],
            key=lambda f:(f['incidencias_bloqueantes'],-f['regiones_48_potenciales_no_acreditadas']['metal'])),
        'nueve_envases_multicapa':[{'imagen':r['imagen'],'objetos':r['objetos'],'motivo':r['motivo'],'prioridad':r['prioridad']} for r in incs if 'milk-cardboard' in r['motivo'] or 'juice-cardboard' in r['motivo']],
        'advertencia':'Potencial del guardado existente, no regiones nuevas aceptadas. Requiere corrección humana, cierre de incidencias y QA. No se inicia entrenamiento.'}
    publicar(p.parent/'plan_prioridades.json',recibo)
    print('Plan preparado:',len(filas),'imágenes con observaciones;',len(recibo['nueve_envases_multicapa']),'envases multicapa;',len(recibo['opciones_metal']),'opciones pendientes con metal',flush=True)


if __name__=='__main__':main()
