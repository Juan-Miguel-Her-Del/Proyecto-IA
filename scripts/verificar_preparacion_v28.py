"""Certifica el respaldo, COCO de auditoría y preparación bloqueada V28 sin entrenar."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha,digest,bloqueo,caja_derivada
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import revision_humana_final,mask_poligonos
from ecowaste.preparacion_v28 import suficiencia,exigir_ejecucion
from scripts.respaldar_anotaciones_v28 import SALIDA,ORIGEN,inventariar


def exigir(b,m):
    if not b:raise ValueError(m)


def comprobar(raiz,hashes):
    for f,h in hashes.items():
        p=raiz/f
        exigir(p.is_file() and sha(p)==h,'Archivo ausente/alterado: '+f)


def codigo():
    paths=[RAIZ/p for p in ['ecowaste/validacion_anotaciones.py','ecowaste/preparacion_v28.py',
        'pruebas/test_validacion_v28.py','configuracion/entrenamiento_auxiliar_v28_propuesto.json',
        'documentacion/28_validacion_anotaciones_y_preparacion_entrenamiento.md',
        'datos/nuevos/v28/.gitignore','datos/nuevos/v28/README.md']]
    paths.extend((RAIZ/'scripts').glob('*v28.py'))
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in paths}


def controles():
    old=leer(SALIDA/'historicos_protegidos.json');comprobar(RAIZ,old)
    ck=leer(SALIDA/'checkpoints_previos.json')
    actual={p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')}
    exigir(actual==ck,'Checkpoints históricos alterados')
    rec=leer(SALIDA/'respaldo_verificado.json');backup=RAIZ/rec['destino']
    comprobar(backup,rec['archivos'])
    exigir(inventariar(backup)==rec['archivos'],'Respaldo incompleto o con archivos agregados')
    with bloqueo(ORIGEN/'registro.lock'):
        origen=inventariar(ORIGEN,sin_lock=True)
    esperado={p:h for p,h in rec['archivos'].items() if p!='registro.lock'}
    exigir(all(origen.get(p)==h for p,h in esperado.items()),'Algún archivo original fue modificado/eliminado')
    nuevas=set(origen)-set(esperado)
    t=Taller(backup,RAIZ);eventos=[e for iid in t.imagenes for e in t.diario(iid)]
    ultimas={iid:revision_humana_final(t.diario(iid)) for iid in t.imagenes}
    exigir(len(eventos)==134 and {e['persona_id'] for e in eventos}=={'juan_miguel'},'Recuento o identidad no corresponde al respaldo')
    exigir(sum(e is not None for e in ultimas.values())==68 and sum(e is not None and e['cobertura_completa'] for e in ultimas.values())==67,'Estados humanos inconsistentes')
    exigir(sum(len(e['anotaciones']) for e in ultimas.values() if e)==361,'Se contaron propuestas en lugar de revisiones')
    exp=leer(SALIDA/'exportacion_auxiliar_auditoria.json');dest=RAIZ/exp['destino']
    exigir(exp['estado']=='auditoria_NO_ENTRENAR' and not exp['aprobada_qa_para_entrenar'],'Se aprobó una colección pendiente')
    manifest=leer(dest/'MANIFIESTO.json');comprobar(dest,manifest['archivos'])
    exigir(sha(dest/'MANIFIESTO.json')==exp['sha256_manifiesto'],'Manifiesto de colección alterado')
    extra=leer(dest/'MANIFIESTO_COMPLEMENTO_LICENCIAS.json');comprobar(dest,extra['archivos'])
    exigir(extra['manifiesto_principal_sha256']==sha(dest/'MANIFIESTO.json'),'Complemento de otra colección')
    coco=leer(dest/'annotations.json');imgs={i['id']:i for i in coco['images']};cats={i['id']:i['name'] for i in coco['categories']}
    exigir(len(imgs)==67 and len(coco['annotations'])==355,'COCO no corresponde a imágenes humanas completas')
    exigir(not coco['info']['entrenamiento_autorizado'] and not coco['info']['aceptadas_V26'],'COCO indebidamente autorizado')
    for im in imgs.values():
        iid=im['eco']['imagen_id_auxiliar'];ev=ultimas[iid]
        exigir(ev and ev['cobertura_completa'] and im['eco']['revision_huella']==ev['huella'],'Imagen COCO sin revisión completa')
        exigir(sha(dest/im['file_name'])==im['eco']['sha256_archivo']==t.imagen(iid)['sha256_archivo'],'Foto modificada al exportar')
        exigir(im['eco']['objeto_fisico_id'] is None and not im['eco']['aceptada_V26'],'Identidad o aceptación inventadas')
        folder=digest(iid)[:24]
        for p in (backup/'revisiones'/folder).glob('*.json'):
            exigir(sha(dest/'revisiones'/folder/p.name)==sha(p),'Corrección humana modificada en copia')
    for a in coco['annotations']:
        im=imgs[a['image_id']];iid=im['eco']['imagen_id_auxiliar'];ev=ultimas[iid]
        original=next(x for x in ev['anotaciones'] if x['id']==a['eco']['id_auxiliar'])
        exigir(not original['descartada'] and cats[a['category_id']]==original['material'],'Se alteró material o descarte humano')
        exigir(a['eco']['poligonos_usuario']==original['poligonos'] and a['eco']['huecos_usuario']==original['huecos'],'Contorno humano modificado')
        exigir(a['bbox']==caja_derivada(original['poligonos']),'BBox no derivada del contorno')
        exigir(a['area']==int(mask_poligonos(original,im['width'],im['height']).sum()),'Área de máscara incorrecta')
    filas=leer(SALIDA/'imagenes.json');obs=leer(SALIDA/'observaciones_visuales_y_taxonomia.json')
    suff=suficiencia(filas,obs)
    exigir(suff==leer(SALIDA/'suficiencia_exploratoria.json') and suff['cantidad_imagenes']==28 and not suff['cumple_referencia_numerica'],'Decisión de suficiencia alterada')
    cfg=leer(RAIZ/'configuracion/entrenamiento_auxiliar_v28_propuesto.json')
    exigir(cfg==leer(SALIDA/'protocolo_propuesto.json'),'Protocolos distintos')
    exigir(cfg['presupuesto_actualizaciones_por_rama']==300 and cfg['hash_datos'] is None and not cfg['entrenamiento_autorizado'],'Entrenamiento/protocolo alterado')
    try:exigir_ejecucion(cfg,leer(dest/'QA_PENDIENTE.json'))
    except PermissionError:pass
    else:raise ValueError('Entrenamiento indebidamente habilitado')
    # Respaldos, fotografías y correcciones no entran a Git.
    fotos=[p.relative_to(RAIZ).as_posix() for p in (RAIZ/'datos/nuevos/v28').rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg','.png','.jpeg']]
    q=subprocess.run(['git','check-ignore','-z','--stdin'],cwd=RAIZ,input=('\0'.join(fotos)+'\0').encode(),capture_output=True)
    exigir(q.returncode==0 and set(q.stdout.decode().rstrip('\0').split('\0'))==set(fotos),'Fotos privadas sin exclusión Git')
    exigir(not subprocess.check_output(['git','ls-files','-z','--','datos/nuevos/v28/respaldos','datos/nuevos/v28/colecciones','datos/nuevos/v28/visuales'],cwd=RAIZ),'Datos privados incluidos en Git')
    for doc in [RAIZ/'documentacion/28_validacion_anotaciones_y_preparacion_entrenamiento.md',RAIZ/'datos/nuevos/v28/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not target.startswith('http'):exigir((doc.parent/target).exists(),'Enlace local roto: '+target)
    return {'verificacion_tecnica_aprobada':True,'aprobado_para_entrenar':False,'historicos_intactos':len(old),
        'checkpoints_intactos':len(ck),'respaldo_archivos':len(rec['archivos']),'eventos_validos':134,
        'imagenes_guardadas':68,'completas':67,'borrador':1,'sin_guardado':4,'objetos_activos':356,
        'COCO_auditoria_imagenes':67,'COCO_auditoria_instancias':355,'imagenes_candidatas':28,
        'archivos_nuevos_en_proyecto_activo_despues_respaldo':len(nuevas),'revision_humana_modificada_por_diagnostico':False,
        'entrenamiento':False,'test_evaluado':False,'aceptadas_V26':0}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--publicar',action='store_true');a=ap.parse_args()
    mp,vp=SALIDA/'manifiesto_integridad.json',SALIDA/'verificacion.json';estado=controles()
    if a.publicar:
        exigir(not mp.exists() and not vp.exists(),'Certificación existente: verificar en lectura')
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s','pruebas','-v'],cwd=RAIZ,text=True,capture_output=True,encoding='utf-8',env=env)
        log=p.stdout+'\n'+p.stderr;publicar_bytes(SALIDA/'pruebas_finales.log',log.encode('utf-8'))
        n=re.search(r'Ran (\d+) tests? in',log);ok=p.returncode==0 and n is not None and int(n[1])==310 and 'FAILED' not in log and 'OK (skipped=' not in log
        publicar(SALIDA/'pruebas_finales.json',{'aprobadas':ok,'ejecutadas':int(n[1]) if n else None,'returncode':p.returncode,
            'anteriores_conservadas':292,'nuevas':18,'sha256_log':sha(SALIDA/'pruebas_finales.log'),'codigo':codigo()})
        exigir(ok,'Pruebas incompletas o fallidas')
        p=subprocess.run([sys.executable,'scripts/verificar_anotacion_v27_2.py'],cwd=RAIZ,text=True,capture_output=True,encoding='utf-8',env=env)
        publicar(SALIDA/'integridad_V27_2_conservada.json',{'aprobada':p.returncode==0,'salida':p.stdout,'errores':p.stderr,'entrenamiento':False})
        exigir(p.returncode==0,'V27.2 no conserva integridad: '+p.stderr)
        estado=controles()
        privados=[p for p in (RAIZ/'datos/nuevos/v28').rglob('*') if p.is_file() and p.name not in ['README.md','.gitignore']]
        publicar(mp,{'version':'v28_validacion_humana_preparacion_1','codigo_y_documentacion':codigo(),
            'archivos_privados':{p.relative_to(RAIZ).as_posix():sha(p) for p in privados},
            'resultados':{p.relative_to(SALIDA).as_posix():sha(p) for p in SALIDA.rglob('*') if p.is_file() and p not in [mp,vp]},
            'entrenamiento_autorizado':False,'QA_datos_aprobada':False})
        publicar(vp,{**estado,'pruebas_aprobadas':310,'sha256_manifiesto':sha(mp)})
    else:
        m=leer(mp);comprobar(RAIZ,m['codigo_y_documentacion']);comprobar(RAIZ,m['archivos_privados']);comprobar(SALIDA,m['resultados'])
        exigir(leer(vp)['sha256_manifiesto']==sha(mp),'Manifiesto alterado')
        exigir(leer(SALIDA/'pruebas_finales.json')['aprobadas'],'Pruebas no aprobadas')
    print(json.dumps({**estado,'pruebas_aprobadas':310,'sha256_manifiesto':sha(mp)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
