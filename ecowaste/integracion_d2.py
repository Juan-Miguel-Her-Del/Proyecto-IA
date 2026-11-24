"""D2 del equipo + clasificación seleccionable + asistente propio. Solo inferencia."""
import hashlib,json,math,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from ecowaste.datasets import CLASES,letterbox
from ecowaste.clasificador import cargar_clasificador
from ecowaste.asistente import Asistente
from ecowaste.incorporacion import sha
from ecowaste.vision_equipo.localizador import NeuralDetector
from ecowaste.vision_equipo.modelo import VisionCNN
from ecowaste.vision_equipo.compacto import CompactCNN,CompactTransform
from ecowaste.vision_equipo.preprocesado import ImageTransform,crop_box

RAIZ=Path(__file__).resolve().parents[1]
MAPEO={'plastico':'plastico','papel':'papel','carton':'carton','vidrio':'vidrio','metal':'metal','organicos':'organico','organico':'organico'}
MODOS=('transformer','refinado','escena','combinado','historico')
PESOS={'refinado':'classifier_materials.pth','escena':'classifier_scene.pth','historico':'classifier_B.pth'}

def leer_config(path=None):
    p=Path(path) if path else RAIZ/'configuracion/integracion_d2.json';c=json.loads(p.read_text(encoding='utf8'))
    if c['version']!='ecowaste_d2_equipo_transformer_1' or c['entrenamiento']:raise ValueError('Configuración de integración inválida')
    if c['vistas']!=[[512,False],[640,False],[512,True]] or c['umbral_deteccion']!=.27 or c['nms_iou']!=.45:raise ValueError('Esta versión conserva las tres vistas y umbrales D2 verificados')
    if c['peso_refinado']!=.65 or c['peso_escena']!=.35 or c['margen_recorte']!=.12:raise ValueError('Política de mezcla/recorte incompatible')
    if not isinstance(c['cpu_threads'],int) or not 1<=c['cpu_threads']<=4 or not 0<=c['baja_confianza_clasificador']<=1:raise ValueError('Hilos o confianza inválidos')
    return c

def permutacion(clases):
    try:mapeadas=[MAPEO[c] for c in clases]
    except KeyError as e:raise ValueError('Material externo desconocido') from e
    if len(mapeadas)!=6 or len(set(mapeadas))!=6 or set(mapeadas)!=set(CLASES):raise ValueError('Correspondencia de clases incompleta/duplicada')
    return [mapeadas.index(c) for c in CLASES]

def verificar_pesos(cfg):
    manifest=json.loads((RAIZ/cfg['manifest']).read_text(encoding='utf8'));res={}
    for n in ['detector_D2.pth',*PESOS.values()]:
        p=RAIZ/'checkpoints/vision_equipo'/n;m=manifest[n]
        if m.get('trained_in_project') is not True or m.get('foreign_pretrained_weights') is not False or sha(p)!=m['sha256']:raise ValueError('Pesos no verificados: '+n)
        res[n]=m['sha256']
    if sha(RAIZ/cfg['detector'])!=manifest['detector_D2.pth']['sha256']:raise ValueError('La configuración apunta a otro checkpoint D2')
    if sha(RAIZ/cfg['transformer'])!=cfg['sha256_transformer']:raise ValueError('Transformer propio alterado')
    res['transformer']=cfg['sha256_transformer'];res['asistente']=sha(RAIZ/cfg['asistente']);return res

def preparar_transformer(rgb):
    # RGB [0,1], letterbox224; el modelo aplica (x-.5)/.5 internamente.
    return letterbox(Image.fromarray(rgb),224)[0]

class BancoClasificadores:
    def __init__(self,cfg,device='cpu'):
        self.cfg=cfg;self.device=torch.device(device);self.modelos={};self.metadatos={};self.transformaciones={}
    def cargar(self,modo):
        if modo=='combinado':self.cargar('refinado');self.cargar('escena');return
        if modo not in MODOS:raise ValueError('Clasificador desconocido')
        if modo in self.modelos:return
        if modo=='transformer':
            m,ck=cargar_clasificador(RAIZ/self.cfg['transformer'],self.device);t=preparar_transformer;clases=ck['clases']
        else:
            ck=torch.load(RAIZ/'checkpoints/vision_equipo'/PESOS[modo],map_location='cpu',weights_only=True);c=ck['config'];clases=ck['classes'];permutacion(clases)
            norm=c['normalization']
            if any(len(norm[k])!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in norm[k]) for k in ['mean','std']) or min(norm['std'])<=0:raise ValueError('Normalización no válida')
            if ck['architecture']=='VisionCNN-v1' and ck['image_size']==128:
                m=VisionCNN(dropout=c['dropout']);tr=ImageTransform(False,norm)
            elif ck['architecture']=='VisionResidual-v2' and ck['image_size'] in [160,192]:
                m=CompactCNN(ck['image_size'],c['dropout'],c.get('channels',(24,48,96,160)),c.get('pooling','avg'));tr=CompactTransform(c,False)
            else:raise ValueError('Arquitectura de clasificación incompatible')
            m.load_state_dict(ck['model_state_dict'],strict=True);m=m.to(self.device).eval();t=lambda rgb,tr=tr:tr(Image.fromarray(rgb))
        m.requires_grad_(False);self.modelos[modo]=m;self.metadatos[modo]=ck;self.transformaciones[modo]=t
    @torch.inference_mode()
    def probabilidades(self,recortes,modo='transformer'):
        self.cargar(modo)
        if not recortes:return np.empty((0,6),np.float32)
        if modo=='combinado':return .65*self.probabilidades(recortes,'refinado')+.35*self.probabilidades(recortes,'escena')
        outputs=[];ck=self.metadatos[modo];orden=permutacion(ck['clases'] if modo=='transformer' else ck['classes'])
        # Batch1 mantiene el comportamiento original de Vision-IA sobre cada ROI.
        for rgb in recortes:
            x=self.transformaciones[modo](rgb).unsqueeze(0).to(self.device);z=self.modelos[modo](x)
            if not torch.isfinite(z).all():raise RuntimeError('Logits no finitos')
            p=z.softmax(-1)
            if modo!='transformer' and ck['config'].get('inference_hflip',False):p=(p+self.modelos[modo](x.flip(-1)).softmax(-1))/2
            outputs.append(p[0,orden].cpu().numpy())
        result=np.stack(outputs)
        if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5):raise RuntimeError('Probabilidades inválidas')
        return result

class MotorD2:
    def __init__(self,cfg=None,clasificador=None,dispositivo=None):
        self.cfg=dict(cfg or leer_config());self.modo=clasificador or self.cfg['clasificador']
        if self.modo not in MODOS:raise ValueError('Clasificador desconocido')
        torch.set_num_threads(self.cfg['cpu_threads']);cv2.setNumThreads(1);device=dispositivo or self.cfg['dispositivo']
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA no disponible')
        self.device=device;hashes=verificar_pesos(self.cfg)
        self.detector=NeuralDetector(RAIZ/self.cfg['detector'],cpu_threads=self.cfg['cpu_threads'],confidence=self.cfg['umbral_deteccion'],nms_iou=self.cfg['nms_iou'],views=self.cfg['vistas'],device=device)
        self.banco=BancoClasificadores(self.cfg,device);self.banco.cargar(self.modo);self.asistente=None
        self.metadatos={'motor':'d2-transformer' if self.modo=='transformer' else 'd2-vision','detector':'VisionLocaliza-D2','clasificador':self.modo,'vistas':self.cfg['vistas'],'umbral_detector':.27,'nms_iou':.45,'margen_recorte':.12,'dispositivo':device,'precision_numerica':'float32_sin_AMP','checkpoints_sha256':hashes,'seguimiento_de_objetos':False,'limite':'Experimental: material predicho, confianza no calibrada; todos los objetos D2 válidos, sin seguimiento de identidades.'}
    def detectar_rgb(self,rgb):
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<32:raise ValueError('Se requiere RGB uint8 de al menos 32x32')
        inicio=time.perf_counter();raw=self.detector.detect(rgb);dt=time.perf_counter()-inicio;h,w=rgb.shape[:2];boxes=[];recortes=[];validas=[]
        for d in raw:
            b=np.asarray(d.box,dtype=float)
            if not np.isfinite(b).all() or not math.isfinite(d.confidence):raise RuntimeError('D2 produjo valores no finitos')
            if not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise RuntimeError('Caja D2 fuera de límites')
            box=tuple(map(int,d.region.box))
            if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise RuntimeError('Recorte fuera de límites')
            boxes.append(box);recortes.append(rgb[box[1]:box[3],box[0]:box[2]].copy());validas.append(d)
        tc=time.perf_counter();ps=self.banco.probabilidades(recortes,self.modo) if recortes else np.empty((0,6),np.float32);ct=time.perf_counter()-tc;ds=[]
        for i,(d,box,r,p) in enumerate(zip(validas,boxes,recortes,ps)):
            c=int(p.argmax());sc=float(p[c]);low=sc<self.cfg['baja_confianza_clasificador']
            ds.append({'indice_objeto_frame':i,'bbox_xyxy':list(d.box),'bbox_recorte_xyxy':list(box),'material':CLASES[c],'categoria_operativa':'organico' if CLASES[c]=='organico' else 'inorganico','confianza_detector':float(d.confidence),'confianza_clasificador':sc,'puntuacion':float(d.confidence)*sc,'puntuacion_conjunta_no_calibrada':float(d.confidence)*sc,'probabilidades_material':dict(zip(CLASES,map(float,p))),'clasificacion_baja_confianza':low,'requiere_confirmacion_material':low,'etiqueta_instantanea':CLASES[c],'etiqueta_estabilizada':None,'crop_sha256_pixeles':hashlib.sha256(r.tobytes()).hexdigest()})
        return {'detecciones':ds,'ancho':w,'alto':h,'coordenadas':'xyxy_pixeles_imagen_original','tiempos_ms':{'detector':dt*1000,'clasificacion':ct*1000,'total':(time.perf_counter()-inicio)*1000},'estado_detector':'integracion_equipo_experimental','limite':self.metadatos['limite'],'materiales_son_predicciones':True}
    def detectar(self,frame_bgr):
        return self.detectar_rgb(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    def consultar(self,pregunta,resultado):
        if self.asistente is None:self.asistente=Asistente(RAIZ/self.cfg['asistente'])
        todos=resultado['detecciones'];confirmables=[d for d in todos if not d.get('clasificacion_baja_confianza',False)];r=self.asistente.consultar(pregunta,confirmables);bajos=len(todos)-len(confirmables)
        if bajos:r['respuesta']=f'Hay {len(todos)} detecciones; {bajos} tienen material de baja confianza y requieren confirmación visual. '+r['respuesta']
        r.update(origen='detecciones_reales_de_esta_imagen_o_ultimo_frame',objetos_total=len(todos),objetos_baja_confianza=bajos,materiales_usados_en_recomendacion=[d['material'] for d in confirmables],modelo='EcoWasteFormer-Assistant propio',puntuaciones_no_calibradas=True);return r
