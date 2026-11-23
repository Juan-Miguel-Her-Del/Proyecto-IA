"""Cierre de preparación: RLE, fuentes, respaldos y cero aprobaciones simuladas."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import urllib.request
import json
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import numpy as np
from ecowaste.incorporacion import leer,sha,digest,publicar
from ecowaste.auxiliares_publicos import Taller,rle_coco
from ecowaste.ajustes_ia_v28_3 import decodificar,CLASES
from ecowaste.revision_ia_v28_4 import RevisionIA,validar_anotacion
from scripts.anotar_v27_2 import PROYECTO


def main():
    cfg=leer(RAIZ/'configuracion/revision_ia_v28_4.json');res=RAIZ/cfg['resultados'];prep=leer(res/'preparacion.json')
    if (res/'verificacion_preparacion.json').exists():raise FileExistsError('Cierre conservado; no sobrescribir')
    pruebas=sorted((res/'pruebas').glob('*/pruebas.json'))[-1];p=leer(pruebas)
    navegador=sorted((res/'pruebas_navegador').glob('*/prueba_editor.json'))[-1];n=leer(navegador)
    if not p['aprobada'] or p['cantidad_pruebas']<409 or sha(pruebas.parent/'suite.log')!=p['sha256_log']:raise ValueError('Suite pendiente')
    if not n['aprobada'] or len(n['casos'])<13 or n['errores_js'] or n['POST_proyecto_real'] or n['SAM_real_ejecutado']:raise ValueError('Navegador pendiente')
    taller=Taller(PROYECTO,RAIZ);rev=RevisionIA(RAIZ/cfg['proyecto'],RAIZ,taller);evs=rev.eventos()
    if evs:raise ValueError('El usuario ya comenzó: conservar decisiones y crear otro cierre de estado, sin reiniciar el diario')
    if sum(len(taller.diario(i)) for i in taller.imagenes)!=197:raise ValueError('El historial original cambió; revisar antes de certificar')
    for rel,h in prep['archivos_respaldo'].items():
        if sha(PROYECTO/rel)!=h or sha(RAIZ/prep['respaldo']/rel)!=h:raise ValueError('Original/respaldo modificado: '+rel)
    anterior=leer(RAIZ/'resultados/v28_3_correcciones/respaldo.json')
    for rel,h in anterior['archivos'].items():
        if sha(RAIZ/anterior['destino']/rel)!=h:raise ValueError('Respaldo histórico modificado')
    protegidos=leer(RAIZ/'resultados/v28_3_correcciones/historicos_protegidos.json')
    for nombre,h in protegidos.items():
        if sha(RAIZ/nombre)!=h:raise ValueError('Histórico modificado: '+nombre)
    rev.verificar_fuentes();aud=rev.ultima_auditoria()
    if not aud['vigente']:raise ValueError('Auditoría RLE pendiente; esperar al servicio o ejecutar auditar_revision_v28_4.py')
    with urllib.request.urlopen(f"http://127.0.0.1:{cfg['puerto']}/api/config",timeout=15) as r:live=json.load(r)
    if live['proyecto_sha256']!=rev.id or live.get('reinicio_necesario',True) or live['entrenamiento'] or live['GPU_utilizada']:raise ValueError('Servicio no actualizado')
    # La publicación IA se valida, no se acepta. No se crean eventos humanos.
    masks=0;ignores=0
    for path in sorted((RAIZ/cfg['publicacion_IA']).glob('[0-9][0-9].json')):
        fila=leer(path)
        for a in fila['anotaciones']:
            if a.get('descartada') or not a.get('segmentacion_canonica_RLE'):continue
            b=validar_anotacion(a,fila['imagen']);m=decodificar(a['segmentacion_canonica_RLE'])
            if b['segmentacion_canonica_RLE']!=a['segmentacion_canonica_RLE'] or not np.array_equal(m,decodificar(rle_coco(m))):raise ValueError('Máscara IA cambió al validar')
            masks+=1;ignores+=a['material'] not in CLASES
    export,coco=rev.exportar(True);porimagen={im['id']:im for im in coco['images']};estados={}
    for a in coco['annotations']:
        iid=porimagen[a['image_id']]['eco_imagen_id']
        if iid not in estados:estados[iid]={x['id']:x for x in rev.estado(iid,evs)['anotaciones']}
        original=estados[iid][a['eco_objeto_id']]
        if a['segmentation']!=original['segmentacion_canonica_RLE']:raise ValueError('Exportador alteró RLE')
        if a['ignore']!=int(original['material'] not in CLASES):raise ValueError('Exportador perdió ignore')
        if a['eco']['revision_V284'] is not None:raise ValueError('No se deben inventar revisiones')
    if rev.eventos()!=evs or any((rev.path/'sam').glob('*.json')):raise ValueError('Se crearon decisiones o se volvió a segmentar la colección real')
    recibo={'version':'cierre_preparacion_v284_1','fecha_UTC':datetime.now(timezone.utc).isoformat(),
        'aprobada':True,'eventos_originales_intactos':197,'eventos_V284_creados_por_agente':0,
        'archivos_respaldo_actual':len(prep['archivos_respaldo']),'archivos_respaldo_historico':len(anterior['archivos']),
        'historicos_verificados':len(protegidos),'checkpoints_verificados':sum(k.startswith('checkpoints/') for k in protegidos),
        'fuentes_IA_intactas':True,'mascaras_IA_RLE_verificadas':masks,'ignore_IA_verificados':ignores,
        'im_con_propuestas':prep['imagenes_con_propuestas'],'propuestas_grupos':prep['propuestas_grupos'],
        'imagenes_utilizables_iniciales':aud['resumen']['imagenes_utilizables'],'positivos_utilizables_iniciales':aud['resumen']['objetos_utilizables'],
        'positivos_confirmados_V284':aud['resumen']['positivos_confirmados_en_propuestas_V284'],
        'exportacion_inicial':export.relative_to(RAIZ).as_posix(),'COCO_imagenes':len(coco['images']),
        'COCO_positivos':sum(not a['ignore'] for a in coco['annotations']),'COCO_ignore':sum(a['ignore'] for a in coco['annotations']),
        'RLE_exportado_exacto':True,'pruebas':p['cantidad_pruebas'],'casos_navegador':len(n['casos']),
        'backend_en_memoria_sha256':live['backend_en_memoria_sha256'],'url':f"http://127.0.0.1:{cfg['puerto']}/",
        'SAM_real_ejecutado':False,'entrenamiento':False,'test_evaluado':False,'generalizacion_desbloqueada':False,
        'checkpoint_predeterminado_modificado':False,'aceptadas_V26':0,
        'no_certifica':'Exhaustividad visual, verdad del material, independencia física o aceptación formal V26.'}
    publicar(res/'verificacion_preparacion.json',recibo)
    codigo=[RAIZ/'ecowaste/revision_ia_v28_4.py',RAIZ/'configuracion/revision_ia_v28_4.json',
        RAIZ/'pruebas/test_revision_ia_v28_4.py',RAIZ/'documentacion/28_4_revision_ajustes_IA.md']
    codigo+=list((RAIZ/'scripts').glob('*v28_4*'))+list((RAIZ/'interfaz/anotacion_v28_4').glob('*'))
    codigo += [RAIZ/'ecowaste'/x for x in ['incorporacion.py','auxiliares_publicos.py','ajustes_ia_v28_3.py','validacion_anotaciones.py','asistente_sam2.py','datasets.py','poligonos.py','supervision_regional.py']]
    evidencias=[res/'preparacion.json',res/'verificacion_preparacion.json',pruebas,pruebas.parent/'suite.log',navegador,
        rev.path/'catalogo.json',Path(aud['archivo']),Path(aud['archivo']).parent/'imagenes.json',export/'annotations.json',export/'manifiesto.json']
    evidencias+=[RAIZ/cfg['manifiesto_IA'],RAIZ/'resultados/v27_2_anotacion/sam2_pesos.json',RAIZ/'resultados/v27_2_anotacion/sam2_archivos_fuente.json']
    fuentes={p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(codigo+evidencias)) if p.is_file()}
    publicar(res/'manifiesto_integridad_preparacion.json',{'version':'integridad_preparacion_V284_1','archivos':fuentes,
        'proyecto_sha256':rev.id,'archivos_IA_origen':rev.catalogo['archivos_fuente'],
        'pruebas':p['cantidad_pruebas'],'navegador':len(n['casos']),
        'entrenamiento':False,'aceptadas_V26':0,'logs_activos_y_cache_excluidos':True,
        'nota':'Certificado de preparación. Las decisiones posteriores del usuario producen nuevos eventos y auditorías sin sustituir este recibo.'})
    publicar(res/'recibo_integridad.json',{'sha256_manifiesto':sha(res/'manifiesto_integridad_preparacion.json')})
    print(recibo,flush=True)


if __name__=='__main__':main()
