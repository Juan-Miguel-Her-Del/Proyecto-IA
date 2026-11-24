"""Suite completa y regresión contra fuentes originales archivadas, sin importarlas."""
import argparse,ast,hashlib,json,os,re,subprocess,sys,time,threading
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ));OUT=RAIZ/'resultados/integracion_d2'
def publicar(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def suite(registro='pruebas'):
    res={};env=os.environ.copy();env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    for nombre,args in [('suite',[sys.executable,'-m','unittest','discover','-s','pruebas']),('codec',[str(Path.home()/'.venvs/EcoWasteAnotacion/Scripts/python.exe'),'-m','unittest','pruebas.test_datos_y_entrenamiento_v29'])]:
        t=time.perf_counter();r=subprocess.run(args,cwd=RAIZ,capture_output=True,text=True,encoding='utf8',errors='replace',env=env);texto=r.stdout+r.stderr;p=OUT/f'{registro}/{nombre}.log';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf8') as f:f.write(texto)
        n=re.search(r'Ran (\d+) tests',texto);res[nombre]={'returncode':r.returncode,'casos':int(n[1]) if n else 0,'segundos':time.perf_counter()-t,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};print(nombre,res[nombre],flush=True)
    publicar(OUT/(registro+'.json'),{'aprobada':all(r['returncode']==0 for r in res.values()),'resultados':res})
    if any(r['returncode'] for r in res.values()):raise RuntimeError('Falló suite')
def referencia():
    import numpy as np,torch
    from PIL import Image,ImageOps,ImageEnhance,ImageFilter
    from dataclasses import dataclass
    from ecowaste.integracion_d2 import MotorD2,CLASES,permutacion
    from ecowaste.vision_equipo import localizador as loc
    from ecowaste.vision_equipo.preprocesado import letterbox,image_tensor
    from ecowaste.vision_equipo.detector import VisionLocaliza,decode_predictions,ARCHITECTURE
    from ecowaste.vision_equipo.metricas import nms,fuse_detection_views
    from ecowaste.vision_equipo.compacto import CompactCNN
    from ecowaste.vision_equipo.modelo import VisionCNN
    from ecowaste.incorporacion import sha
    import random
    ns=dict(np=np,torch=torch,Image=Image,ImageOps=ImageOps,ImageEnhance=ImageEnhance,ImageFilter=ImageFilter,random=random,Path=Path,threading=threading,time=time,dataclass=dataclass,ARCHITECTURE=ARCHITECTURE,VisionLocaliza=VisionLocaliza,decode_predictions=decode_predictions,letterbox=letterbox,image_tensor=image_tensor,nms=nms,fuse_detection_views=fuse_detection_views,sha256=sha)
    def extraer(file,names):
        s=(OUT/'fuentes'/file).read_text(encoding='utf8');ls=s.splitlines()
        for n in ast.parse(s).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:
                t='\n'.join(ls[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]);t=t.replace('from utils.dataset import sha256','').replace('from training.compact import CompactCNN','from ecowaste.vision_equipo.compacto import CompactCNN');exec(compile(t,'fuente_original_archivada/'+file,'exec'),ns)
    extraer('app/localization.py',['Region','square_box']);extraer('app/detector.py',['crop_box','Detection','NeuralDetector']);extraer('training/data.py',['ImageTransform']);extraer('training/compact.py',['CompactTransform']);extraer('training/prediction.py',['probabilities_from_logits','model_probabilities'])
    m=MotorD2();original=ns['NeuralDetector'](RAIZ/m.cfg['detector'],cpu_threads=2,confidence=.27,nms_iou=.45,views=m.cfg['vistas']);datos=json.loads((OUT/'cohortes.json').read_text())['d2_val_autorizada'];seleccion=sorted(datos,key=lambda r:r['id_original'])[:4];rows=[]
    for r in seleccion:
        with Image.open(RAIZ/r['ruta']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        a=original.detect(rgb);b=m.detector.detect(rgb)
        if len(a)!=len(b):raise AssertionError('D2 cambió número de objetos')
        for x,y in zip(a,b):
            if x.box!=y.box or x.confidence!=y.confidence or x.region.box!=y.region.box:raise AssertionError('D2 no reproduce la fuente original')
        rows.append({'id':r['id'],'detecciones':len(a),'cajas_puntuaciones_y_recortes_exactos':True})
        crops=[rgb[d.region.box[1]:d.region.box[3],d.region.box[0]:d.region.box[2]].copy() for d in b]
        for modo in ['refinado','escena','historico']:
            m.banco.cargar(modo);ck=m.banco.metadatos[modo];c=ck['config'];tr=ns['ImageTransform'](False,c['normalization']) if modo!='escena' else ns['CompactTransform'](c,False)
            for crop in crops:
                x=tr(Image.fromarray(crop)).unsqueeze(0);y=m.banco.transformaciones[modo](crop).unsqueeze(0)
                if not torch.equal(x,y):raise AssertionError('Cambió preprocesado original')
                with torch.inference_mode():p=ns['model_probabilities'](m.banco.modelos[modo],x,c)[0,permutacion(ck['classes'])].numpy()
                q=m.banco.probabilidades([crop],modo)[0]
                if not np.array_equal(p,q):raise AssertionError('Cambió clasificación original')
        if crops:
            a=m.banco.probabilidades(crops,'combinado');b=.65*m.banco.probabilidades(crops,'refinado')+.35*m.banco.probabilidades(crops,'escena')
            if not np.array_equal(a,b):raise AssertionError('Mezcla no corresponde 65/35')
    blancos={}
    for nombre,color in [('negro',0),('blanco',255),('gris',128)]:blancos[nombre]=len(m.detector.detect(np.full((384,512,3),color,np.uint8)))
    publicar(OUT/'regresion_original.json',{'aprobada':True,'imagenes_reales':rows,'imagenes_diagnosticas_sin_objetos_reales':blancos,'fuentes':'Snapshots dentro de EcoWasteFormer; exec en memoria, ninguna importación o escritura en proyecto ajeno','precision':'CPU float32; identidad exacta observada en estas fotos, no garantía universal entre backends','todas_las_detecciones':True})
    print('Regresión original aprobada',rows,blancos,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--fase',choices=['suite','referencia'],required=True);p.add_argument('--registro',default='pruebas');a=p.parse_args()
    if not a.registro.isidentifier():p.error('Nombre de registro inválido')
    suite(a.registro) if a.fase=='suite' else referencia()
if __name__=='__main__':main()
