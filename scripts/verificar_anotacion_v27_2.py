"""Certifica V27.2 y conserva el estado inicial; valida diarios futuros por separado."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha,digest,poligonos_validos
from ecowaste.propuestas_publicas import imagen_verificada
from ecowaste.auxiliares_publicos import Taller,VERSION
from scripts.fuentes_v27_2 import SALIDA,DATOS
from scripts.anotar_v27_2 import PROYECTO


def exigir(v,m):
    if not v:raise ValueError(m)


def comprobar(raiz,hashes):
    for f,h in hashes.items():
        p=raiz/f
        exigir(p.is_file() and sha(p)==h,'Archivo alterado/ausente: '+f)


def codigo():
    paths=[RAIZ/p for p in ['ecowaste/auxiliares_publicos.py','ecowaste/asistente_sam2.py',
        'configuracion/auxiliares_publicos_v27_2.json','configuracion/exploratorio_auxiliar_v27_2.json',
        'esquemas/auxiliares_publicos_v27_2.json','documentacion/27_2_datasets_y_anotacion_asistida.md',
        'datos/nuevos/v27_2/README.md','datos/nuevos/v27_2/.gitignore','scripts/ABRIR_ANOTADOR_V27_2.cmd']]
    paths.extend((RAIZ/'scripts').glob('*v27_2.py'));paths.extend((RAIZ/'pruebas').glob('*v27_2.py'))
    paths.extend(p for p in (RAIZ/'interfaz/anotacion_v27_2').iterdir() if p.is_file())
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in paths}


def controles():
    old=leer(SALIDA/'historicos_protegidos.json');comprobar(RAIZ,old)
    ck=leer(SALIDA/'checkpoints_previos.json')
    actual={p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')}
    exigir(actual==ck,'Checkpoints históricos alterados')
    t=Taller(PROYECTO,RAIZ)
    exigir(len(t.imagenes)==72 and sum(len(i['anotaciones']) for i in t.imagenes.values())==256,'Base del piloto incorrecta')
    exigir(sum(i['fuente']=='wsc' and i['split_original']=='valid' for i in t.imagenes.values())==12,'WSC: variantes no autorizadas')
    exigir(not any(i['fuente'] in ['dwsd','realwaste'] for i in t.imagenes.values()),'Fuente excluida en tareas activas')
    for i in t.imagenes.values():
        meta=imagen_verificada(t.archivo_imagen(i['id']))
        exigir(all(i[k]==v for k,v in meta.items()),'Imagen del piloto o metadata alterada')
    humanos=sum(len(t.diario(i)) for i in t.imagenes)
    cfg=leer(RAIZ/'configuracion/auxiliares_publicos_v27_2.json')
    exigir(cfg['version']==VERSION and cfg['dispositivo']=='cpu','Contrato/backend incompatible')
    for k in ['entrenamiento_autorizado','evaluacion_autorizada','generalizacion_desbloqueada','aceptacion_formal_V26_implementada_aqui']:
        exigir(cfg[k] is False,'Se amplió autorización: '+k)
    ex=leer(RAIZ/'configuracion/exploratorio_auxiliar_v27_2.json')
    exigir(not ex['entrenamiento_autorizado'] and not ex['evaluacion_autorizada'] and not ex['coleccion_congelada'],'Se habilitó experimento')
    for n,h in [('wsc','2f49ef8a3736ee44fbd451143c9b21e6d3745f6507bc624245c4a85e64197925'),
                ('dwsd','fb9229c330ed253f4fb7c18bd46dddebe4df6f8d3f602747f505ff7601a11737')]:
        exigir(sha(DATOS/n/'original.zip')==h,'ZIP distinto al catálogo '+n)
    fuente=leer(SALIDA/'sam2_archivos_fuente.json')
    comprobar(RAIZ/'datos/nuevos/v27_2/modelos/fuente',fuente['archivos'])
    exigir(sha(RAIZ/'datos/nuevos/v27_2/modelos/sam2.1_hiera_tiny.pt')==leer(SALIDA/'sam2_pesos.json')['sha256'],'Pesos SAM alterados')
    indice=leer(PROYECTO/'PROPUESTAS_IA_INICIALES.json')
    exigir(indice['proyecto_sha256']==t.base['huella'],'Índice de otro proyecto')
    n=0
    for iid,por_a in indice['por_imagen'].items():
        im=t.imagen(iid)
        for aid,pid in por_a.items():
            p=leer(PROYECTO/'propuestas_ia'/(pid+'.json'))
            exigir(p['imagen_id']==iid and p['proyecto_sha256']==t.base['huella'] and not p['aceptada_V26'],'Propuesta de otra imagen o aceptada')
            exigir(p['huella']==digest({k:v for k,v in p.items() if k!='huella'}),'Propuesta SAM alterada')
            poligonos_validos(p['resultado']['poligonos'],im['width'],im['height']);n+=1
    exigir(n==86 and len(leer(SALIDA/'piloto_sam2_cpu.json')['incidencias'])==11,'Resultados SAM incompatibles')
    ui=leer(SALIDA/'prueba_interfaz_revision02.json')
    comprobar(RAIZ,ui['hashes_codigo'])
    exigir(ui['aprobada'] and ui['fixture_descartable_eliminado'] and not ui['errores_JS']
           and not ui['revisiones_humanas_simuladas_en_piloto_real'],'Interfaz no certificada')
    exigir(sha(RAIZ/ui['screenshot_local_no_publicar'])==ui['screenshot_sha256'],'Captura alterada')
    snap=leer(SALIDA/'estado_entrega_piloto.json')
    exigir(snap['imagenes_activas']==72 and snap['propuestas_publicas']==256 and snap['con_poligonos']==245
           and snap['solo_caja_pendiente']==11 and snap['revisiones_humanas']==0 and snap['aceptadas_V26']==0,'Inventario inicial alterado')
    ignorar=[i['archivo'] for i in t.imagenes.values()]
    ignorar.extend(p.relative_to(RAIZ).as_posix() for p in (DATOS/'contactos').glob('*') if p.is_file())
    ignorar.extend(['datos/nuevos/v27_2/modelos/sam2.1_hiera_tiny.pt',PROYECTO.relative_to(RAIZ).as_posix()+'/PROYECTO.json'])
    q=subprocess.run(['git','check-ignore','-z','--stdin'],cwd=RAIZ,input=('\0'.join(ignorar)+'\0').encode(),capture_output=True)
    exigir(q.returncode==0 and set(q.stdout.decode().rstrip('\0').split('\0'))==set(ignorar),'Archivos privados sin exclusión Git')
    tracked=subprocess.check_output(['git','ls-files','-z','--','datos/nuevos/v27_2/externos','datos/nuevos/v27_2/modelos','datos/nuevos/v27_2/proyecto'],cwd=RAIZ)
    exigir(not tracked,'Fotos/modelos/correcciones incluidos en Git')
    for doc in [RAIZ/'documentacion/27_2_datasets_y_anotacion_asistida.md',RAIZ/'datos/nuevos/v27_2/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not target.startswith('http'):exigir((doc.parent/target).exists(),'Enlace local roto: '+target)
    return {'aprobada':True,'historicos_sha256_intactos':len(old),'checkpoints_estadisticas_intactas':len(ck),
            'imagenes_piloto':72,'propuestas_originales':256,'poligonos_preliminares':245,'solo_caja':11,
            'propuestas_SAM_ofrecidas':86,'revisiones_humanas_actuales':humanos,'aceptadas_V26':0,
            'cuotas_train_pendientes':1550,'generalizacion_pendiente':600,'organicos_utilizables_pendientes':300,
            'entrenamiento':False,'test_original_evaluado':False,'generalizacion_desbloqueada':False,'GPU_utilizada_por_SAM':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--publicar',action='store_true');args=ap.parse_args()
    mp,vp=SALIDA/'manifiesto_integridad.json',SALIDA/'verificacion.json'
    estado=controles()
    if args.publicar:
        exigir(not mp.exists() and not vp.exists(),'Certificado existente: verificar en lectura')
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}
        r=subprocess.run([sys.executable,'-m','unittest','discover','-s','pruebas','-v'],cwd=RAIZ,
                         capture_output=True,text=True,encoding='utf-8',env=env)
        log=r.stdout+'\n'+r.stderr;publicar_bytes(SALIDA/'pruebas_finales.log',log.encode('utf-8'))
        m=re.search(r'Ran (\d+) tests? in',log)
        bien=r.returncode==0 and m is not None and int(m[1])==292 and not re.search(r'FAILED|OK \(skipped=',log)
        publicar(SALIDA/'pruebas_finales.json',{'aprobadas':bien,'ejecutadas':int(m[1]) if m else None,
                 'returncode_python':r.returncode,'previas_conservadas':261,'nuevas':31,'sha256_log':sha(SALIDA/'pruebas_finales.log'),
                 'hashes_codigo':codigo()})
        exigir(bien,'Pruebas incompletas/fallidas: consultar pruebas_finales.log')
        previo=subprocess.run([sys.executable,'scripts/verificar_adquisicion_v27.py'],cwd=RAIZ,text=True,
                              capture_output=True,encoding='utf-8',env=env)
        publicar(SALIDA/'integridad_V27_conservada.json',{'aprobada':previo.returncode==0,'returncode':previo.returncode,
                 'salida':previo.stdout,'errores':previo.stderr,'entrenamiento':False,'test_evaluado':False})
        exigir(previo.returncode==0,'Integridad V27 no conservada: '+previo.stderr)
        js=subprocess.run(['node','--check','interfaz/anotacion_v27_2/app.js'],cwd=RAIZ,text=True,capture_output=True)
        exigir(js.returncode==0,'JavaScript no válido: '+js.stderr)
        publicar(SALIDA/'javascript_sintaxis.json',{'aprobada':True,'returncode':js.returncode})
        estado=controles()
        privados=[p for raiz in [DATOS,RAIZ/'datos/nuevos/v27_2/modelos'] for p in raiz.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        privados.extend([PROYECTO/'PROYECTO.json',PROYECTO/'PROPUESTAS_IA_INICIALES.json',PROYECTO.parent/'PROYECTO.json'])
        privados.extend((PROYECTO/'propuestas_ia').glob('*.json'))
        publicar(mp,{'version':VERSION,'codigo_y_documentacion':codigo(),
            'privados_fijos_sha256':{p.relative_to(RAIZ).as_posix():sha(p) for p in privados},
            'resultados':{p.relative_to(SALIDA).as_posix():sha(p) for p in SALIDA.rglob('*') if p.is_file() and p not in [mp,vp]},
            'diarios_usuario':'mutables_por_adicion_fuera_del_manifiesto; verificar_su_cadena',
            'entrenamiento_autorizado':False,'test_evaluado':False})
        publicar(vp,{**estado,'pruebas_aprobadas':292,'integridad_V27_aprobada':True,'sha256_manifiesto':sha(mp)})
    else:
        m=leer(mp);comprobar(RAIZ,m['codigo_y_documentacion']);comprobar(RAIZ,m['privados_fijos_sha256']);comprobar(SALIDA,m['resultados'])
        exigir(leer(vp)['sha256_manifiesto']==sha(mp),'Manifiesto alterado')
        exigir(leer(SALIDA/'pruebas_finales.json')['aprobadas'] and leer(SALIDA/'integridad_V27_conservada.json')['aprobada'],'Certificados no aprobados')
    print(json.dumps({**estado,'pruebas_aprobadas':292,'integridad_V27_aprobada':True,'sha256_manifiesto':sha(mp)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
