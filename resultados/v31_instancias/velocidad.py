"""Latencia local comparable en una foto prefijada, no benchmark de webcam."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,statistics
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ));OUT=Path(__file__).resolve().parent
import cv2,torch
from ecowaste.video import MotorVideo
from ecowaste.video_instancias_v31 import MotorInstanciasV31
from ecowaste.backend_numerico import aplicar_backend,politica,registro_efectivo
from ecowaste.ablacion_fondo import huella
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
def main():
    foto=leer(OUT/'seleccion_visual_previa.json')['imagenes'][1];p=RAIZ/foto['ruta']
    if sha(p)!=foto['sha256']:raise ValueError('Foto alterada')
    frame=cv2.imread(str(p));res={};torch.set_num_threads(4)
    with aplicar_backend(politica('completo')):
        for n in ['Original','A','B']:
            cp=RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt' if n=='Original' else RAIZ/f'checkpoints/v31_instancias/{n}/{leer(OUT/f"metricas_{n}.json")["mejor_paso_por_map50_desarrollo_1v"]:04d}.pt'
            res[n]={}
            for v in [1,5]:
                m=(MotorVideo if n=='Original' else MotorInstanciasV31)(RAIZ,cp,vistas=v,umbral=.35)
                for _ in range(5):m.detectar(frame)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();lat=[];h=[]
                for _ in range(20):
                    torch.cuda.synchronize();t=time.perf_counter();r=m.detectar(frame);torch.cuda.synchronize();lat.append((time.perf_counter()-t)*1000);h.append(huella(r['detecciones']))
                res[n][str(v)]={'ms_mediana':statistics.median(lat),'ms_media':statistics.mean(lat),'ms_min':min(lat),'ms_max':max(lat),'latencias_ms':lat,'pico_gpu_bytes':torch.cuda.max_memory_allocated(),'detecciones':len(r['detecciones']),'predicciones_estables_20':len(set(h))==1,'checkpoint_sha256':sha(cp)}
                del m;torch.cuda.empty_cache();print(n,v,'latencia mediana',res[n][str(v)]['ms_mediana'],flush=True)
        publicar(OUT/'velocidad.json',{'resultados':res,'foto_prefijada':foto,'calentamiento':5,'repeticiones':20,'umbral_visualizacion':.35,'umbral_propuestas':.05,'incluye':'conversión de frame, letterbox, forward AMP, cajas y postproceso; excluye lectura desde disco/dibujo/captura de cámara','camara_fisica':False,'limitacion':'una sola foto repetida; no representa distribución de escenas ni rendimiento webcam completo','backend':registro_efectivo()})
if __name__=='__main__':main()
