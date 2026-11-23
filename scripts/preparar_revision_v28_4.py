"""Congelar fuentes existentes y agrupar propuestas: NO ejecutar segmentación."""
from copy import deepcopy
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha,publicar,digest
from ecowaste.auxiliares_publicos import Taller
from ecowaste.ajustes_ia_v28_3 import canonizar
from ecowaste.revision_ia_v28_4 import VERSION,construir_unidades,RevisionIA,sello
from scripts.anotar_v27_2 import PROYECTO
from scripts.respaldar_anotaciones_v28 import respaldar
from ecowaste.correcciones_v28_3 import auditorias_verificadas


def main():
    cfg=leer(RAIZ/'configuracion/revision_ia_v28_4.json');destino=RAIZ/cfg['proyecto'];res=RAIZ/cfg['resultados']
    if destino.exists():raise FileExistsError('Proyecto preparado; conservarlo, no sobrescribir')
    if cfg['version']!=VERSION or cfg['entrenamiento_autorizado'] or cfg['aceptacion_V26']:raise ValueError('Protocolo incompatible')
    prep=leer(RAIZ/cfg['preparacion_IA']);manifest=leer(RAIZ/cfg['manifiesto_IA'])
    for p,h in manifest['archivos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Fuente IA alterada: '+p)
    backup=RAIZ/'datos/nuevos/v28_4/respaldos'/sello()/'proyecto_humano'
    hashes=respaldar(PROYECTO,backup);t=Taller(backup,RAIZ)
    n=sum(len(t.diario(i)) for i in t.imagenes)
    if n!=197:raise ValueError('El historial ya cambió; conservar respaldo y revisar la nueva base antes de preparar')
    auditorias=auditorias_verificadas(RAIZ/'resultados/v28_3_correcciones/auditorias');ap=auditorias[-1]
    heredadas={r['imagen']:r for r in leer(ap.parent/'imagenes.json')}
    porid={r['imagen']:r for r in prep['imagenes']};filas=[]
    fuentes={cfg['manifiesto_IA']:sha(RAIZ/cfg['manifiesto_IA']),cfg['preparacion_IA']:sha(RAIZ/cfg['preparacion_IA']),
             ap.relative_to(RAIZ).as_posix():sha(ap),(ap.parent/'imagenes.json').relative_to(RAIZ).as_posix():sha(ap.parent/'imagenes.json')}
    for iid,im in t.imagenes.items():
        tarea=t.tarea(iid);ev=next((e for e in reversed(t.diario(iid)) if e['modo']=='asistida'),None)
        base=[canonizar(a,im['width'],im['height']) for a in tarea['anotaciones']]
        completa=bool(ev and ev['cobertura_completa']);advertencias=[];unidades=[];source=None
        if iid in porid:
            f=porid[iid];snapshot=leer(RAIZ/f['archivo']);source=(RAIZ/cfg['publicacion_IA'])/f"{f['indice']:02d}.json";ia=leer(source)
            if ia['revision_humana_base']!=(ev or {}).get('huella'):raise ValueError('Fuente humana obsoleta: '+iid)
            unidades=construir_unidades(base,ia['anotaciones'],im,completa)
            advertencias=[{'motivo':v['motivo'],'objetos':v['objetos']} for v in snapshot['incidencias']]
            advertencias+=ia['observaciones']
            if im['fuente']=='hpi':advertencias.append('Comida pequeña: comprobar piezas omitidas, mezcla y granularidad de instancias.')
            advertencias.append('Envases multicapa, vegetación, oclusión y material incierto requieren motivo e ignore cuando no se pueden resolver; aceptar IA no confirma la cobertura.')
            fuentes[source.relative_to(RAIZ).as_posix()]=sha(source);fuentes[f['archivo']]=sha(RAIZ/f['archivo'])
        heredada=bool(iid not in porid and heredadas[iid]['candidata_exploratoria'] and heredadas[iid]['revision_huella']==(ev or {}).get('huella'))
        filas.append({'imagen':{k:deepcopy(v) for k,v in im.items() if k!='anotaciones'},'base':base,
            'base_completa':completa,'revision_humana_base':(ev or {}).get('huella'),'unidades':unidades,
            'advertencias':advertencias,'fuente_sha256':sha(source) if source else None,
            'candidata_heredada_V283':heredada})
        print(iid,'·',len(unidades),'propuestas/grupos',flush=True)
    catalogo={'version':VERSION,'version_IA_origen':str(cfg['publicacion_IA']),
        'publicacion_IA_sha256':sha(RAIZ/cfg['manifiesto_IA']),'archivos_fuente':fuentes,
        'proyecto_humano_sha256':t.base['huella'],'eventos_humanos_originales':n,
        'personas_historicas':sorted({e['persona_id'] for i in t.imagenes for e in t.diario(i)}),
        'imagenes':filas,'aceptacion_V26':False,'entrenamiento_autorizado':False}
    catalogo['huella']=digest(catalogo);publicar(destino/'catalogo.json',catalogo)
    publicar(res/'preparacion.json',{'version':VERSION,'proyecto':cfg['proyecto'],'huella':catalogo['huella'],
        'respaldo':backup.relative_to(RAIZ).as_posix(),'archivos_respaldo':hashes,'eventos_originales':n,
        'propuestas_grupos':sum(len(f['unidades']) for f in filas),'imagenes_con_propuestas':sum(bool(f['unidades']) for f in filas),
        'heredadas_candidatas':sum(f['candidata_heredada_V283'] for f in filas),'nuevas_decisiones_humanas':0,
        'SAM_ejecutado':False,'entrenamiento':False})
    rev=RevisionIA(destino,RAIZ,Taller(PROYECTO,RAIZ));p,q=rev.auditar()
    publicar(res/'auditoria_inicial.json',{'archivo':p.relative_to(RAIZ).as_posix(),'sha256':sha(p),'resumen':q})
    print(q,flush=True)


if __name__=='__main__':main()
