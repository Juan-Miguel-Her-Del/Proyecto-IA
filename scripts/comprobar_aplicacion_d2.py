"""Casos reales, asistente y despliegue aislado; sin entrenar ni abrir test."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ));OUT=RAIZ/'resultados/integracion_d2';PRIV=RAIZ/'datos/nuevos/integracion_d2'
from ecowaste.integracion_d2 import MotorD2
from ecowaste.aplicacion_integrada import fotografia,guardar_json
from ecowaste.incorporacion import sha
def leer(p):return json.loads(p.read_text(encoding='utf8'))
def casos():
    co=leer(OUT/'cohortes.json')['d2_val_autorizada'];rs={}
    for sp in ['train','val']:
        for l in (RAIZ/f'datos/procesados/deteccion/{sp}.jsonl').read_text(encoding='utf8').splitlines():r=json.loads(l);rs[r['id']]=r
    selected={};none=[]
    for k,meta in enumerate(co):
        r=rs[meta['id']];bs=r['cajas']
        if len(bs)==1:selected.setdefault('un_objeto_GT',meta)
        if len(bs)>=3:selected.setdefault('varios_objetos_GT',meta)
        if any(min(b['bbox'][0]/r['ancho'],b['bbox'][1]/r['alto'],1-(b['bbox'][0]+b['bbox'][2])/r['ancho'],1-(b['bbox'][1]+b['bbox'][3])/r['alto'])<=.02 for b in bs):selected.setdefault('objetos_cerca_borde',meta)
        centers=[((b['bbox'][0]+b['bbox'][2]/2)/r['ancho'],(b['bbox'][1]+b['bbox'][3]/2)/r['alto']) for b in bs]
        if any((x-a)**2+(y-b)**2<=.15**2 for i,(x,y) in enumerate(centers) for a,b in centers[i+1:]):selected.setdefault('objetos_proximos',meta)
        if not leer(OUT/'evaluacion/d2_val_autorizada/capturas'/f'{k:04d}.json')['boxes']:none.append(meta)
    if none:selected['foto_sin_propuestas_D2']=none[0]
    m=MotorD2();results={};multi=None
    for nombre,meta in selected.items():
        out=PRIV/'casos'/nombre;r=fotografia(RAIZ/meta['ruta'],m,out);results[nombre]={'id':meta['id'],'GT_objetos':len(rs[meta['id']]['cajas']),'detecciones':len(r['detecciones']),'directorio_privado':out.relative_to(RAIZ).as_posix(),'sha256_detecciones':sha(out/'detecciones.json')}
        if nombre=='varios_objetos_GT':multi=r
    for modo in ['refinado','escena','combinado','historico']:
        meta=selected['varios_objetos_GT'];out=PRIV/'alternativas'/modo;r=fotografia(RAIZ/meta['ruta'],MotorD2(clasificador=modo),out);results['alternativa_'+modo]={'id':meta['id'],'detecciones':len(r['detecciones']),'directorio_privado':out.relative_to(RAIZ).as_posix(),'sha256_detecciones':sha(out/'detecciones.json')}
    preguntas=['¿Qué residuos aparecen en la imagen?','¿Qué materiales se identificaron?','¿Cuáles son orgánicos o inorgánicos?','¿Cómo debo separarlos?','¿En qué recipiente los deposito?','¿Qué material detectaste?'];respuestas=[{'pregunta':p,**m.consultar(p,multi)} for p in preguntas]
    guardar_json(OUT/'aplicacion_casos_reales.json',{'casos':results,'seleccion':'Primer ID en D2 val autorizada que cumple cantidad/proximidad/borde GT, sin seleccionar aciertos. Ausencia: primer ID sin propuestas; puede contener residuos omitidos, no se afirma escena vacía.','asistente_con_detecciones_reales':respuestas,'GT_en_inferencia':False,'test':False})
    print('Casos reales preparados',results,flush=True)
def portabilidad(suf=''):
    import cv2
    folder=PRIV/('portabilidad/paquete'+suf);folder.mkdir(parents=True,exist_ok=False)
    for q in (RAIZ/'ecowaste').rglob('*.py'):
        dest=folder/q.relative_to(RAIZ);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    for n in ['scripts/ejecutar_ecowaste.py','configuracion/integracion_d2.json','configuracion/arquitectura.json','requirements.txt','requirements-cuda.txt','checkpoints/clasificador_v3_regularizado_mejor.pt','checkpoints/asistente_v1.pt','checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt']:
        q=folder/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(RAIZ/n,q)
    for p in (RAIZ/'checkpoints/vision_equipo').glob('*'):
        if p.is_file():q=folder/'checkpoints/vision_equipo'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    cap=cv2.VideoCapture(str(PRIV/'webcam_fisica_fuera_aislamiento/entrada_capturada.mp4'));ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('No se recuperó foto de video real')
    cv2.imwrite(str(folder/'foto_de_captura_real.png'),frame)
    env=os.environ.copy();env.pop('PYTHONPATH',None);r=subprocess.run([sys.executable,str(folder/'scripts/ejecutar_ecowaste.py'),'--foto',str(folder/'foto_de_captura_real.png'),'--salida',str(folder/'salida'),'--pregunta','¿Qué materiales detectaste?'],cwd=folder,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    guardar_json(OUT/('portabilidad'+suf+'.json'),{'aprobada':True,'raiz_aislada_privada':folder.relative_to(RAIZ).as_posix(),'returncode':r.returncode,'stdout':r.stdout[-3500:],'stderr':r.stderr,'sin_PYTHONPATH_ajeno':True,'sin_importar_carpeta_companero':True,'datos_entrada':'Primer frame de cámara real, dentro del paquete','archivos':{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
    print('Paquete aislado ejecutado correctamente',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--fase',choices=['casos','portabilidad'],required=True);p.add_argument('--revision',default='');a=p.parse_args()
    if a.revision and not a.revision.isidentifier():p.error('Revisión inválida')
    casos() if a.fase=='casos' else portabilidad('_'+a.revision if a.revision else '')
if __name__=='__main__':main()
