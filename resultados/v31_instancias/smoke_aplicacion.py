"""Foto y MP4 con el modelo B real; no prueba cámara ni generalización."""
import json,sys,subprocess,time
from pathlib import Path
import cv2
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ));OUT=Path(__file__).resolve().parent
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
def main():
    datos=leer(OUT/'metricas_B.json');k=datos['mejor_paso_por_map50_desarrollo_1v'];cp=RAIZ/f'checkpoints/v31_instancias/B/{k:04d}.pt';h=sha(cp)
    source=leer(OUT/'seleccion_visual_previa.json')['imagenes'][1];foto=RAIZ/source['ruta']
    if sha(foto)!=source['sha256']:raise ValueError('Foto alterada')
    folder=RAIZ/'datos/nuevos/v31/privado/smoke_aplicacion'
    if folder.exists():raise ValueError('No sobrescribir demo')
    folder.mkdir(parents=True);im=cv2.imread(str(foto));im=cv2.resize(im,(640,480));video=folder/'secuencia_fotografia.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5.,(640,480))
    if not writer.isOpened():raise RuntimeError('Codec MP4 no disponible')
    for _ in range(6):writer.write(im)
    writer.release();res={}
    for tipo,entrada in [('foto',foto),('video',video)]:
        destino=folder/tipo;args=[sys.executable,str(RAIZ/'scripts/detectar_instancias_v31.py'),'--'+tipo,str(entrada),'--checkpoint',str(cp),'--salida',str(destino),'--vistas','1']
        if tipo=='video':args+=['--max-frames','6']
        t=time.perf_counter();run=subprocess.run(args,cwd=RAIZ,capture_output=True,text=True,encoding='utf8',errors='replace');res[tipo]={'codigo_salida':run.returncode,'segundos_proceso':time.perf_counter()-t,'stdout':run.stdout[-3000:],'stderr':run.stderr[-3000:]}
        if run.returncode:raise RuntimeError(res[tipo])
    if sha(cp)!=h:raise ValueError('Inferencia alteró checkpoint')
    publicar(OUT/'aplicacion.json',{'rama':'B','paso':k,'checkpoint':cp.relative_to(RAIZ).as_posix(),'sha256_checkpoint':h,'pruebas':res,'camara_fisica_probada':False,'video_independiente':False,'descripcion_video':'seis repeticiones de una fotografía real, solo prueba lectura/procesamiento/escritura MP4','gt_en_inferencia':False,'adopcion':False,'archivos_privados':{p.relative_to(RAIZ).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}})
    print('Foto y MP4 procesados con B; modelo original intacto',flush=True)
if __name__=='__main__':main()
