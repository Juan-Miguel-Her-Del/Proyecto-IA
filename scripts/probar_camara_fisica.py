"""Prueba breve de cámara real, con límite de tiempo y resultados sin afirmar precisión."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from scripts.entrenar_clasificador import escribir_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indice',type=int,default=0)
    p.add_argument('--max-frames',type=int,default=12)
    p.add_argument('--timeout',type=int,default=30)
    p.add_argument('--nombre',default='prueba_camara_fisica_01')
    a=p.parse_args()
    if a.indice<0 or a.max_frames<=0 or a.timeout<=0 or Path(a.nombre).name!=a.nombre:raise ValueError('Parámetros inválidos')
    carpeta=RAIZ/f'resultados/{a.nombre}';informe=RAIZ/f'resultados/{a.nombre}.json'
    if carpeta.exists() or informe.exists():raise FileExistsError('Prueba existente; usar otro nombre')
    config=RAIZ/'configuracion/arquitectura.json';sha_antes=hashlib.sha256(config.read_bytes()).hexdigest()
    inicio=time.perf_counter()
    cmd=[sys.executable,'scripts/detectar_video.py','--camara',str(a.indice),'--max-frames',str(a.max_frames),'--salida',str(carpeta)]
    resultado={'fecha_utc':datetime.now(timezone.utc).isoformat(),'indice_camara':a.indice,
        'maximo_frames':a.max_frames,'timeout_segundos':a.timeout,'mock_utilizado':False,
        'mostrar_ventana':False,'audio_capturado':False,'inferencia_de_validacion_repetida':False,
        'sha256_seleccion_antes':sha_antes,'test_evaluado':False}
    try:
        ejec=subprocess.run(cmd,cwd=RAIZ,capture_output=True,text=True,encoding='utf-8',errors='replace',
            timeout=a.timeout,env=os.environ|{'PYTHONIOENCODING':'utf-8'})
        resultado.update(codigo_salida=ejec.returncode,stdout=ejec.stdout,stderr=ejec.stderr)
        if ejec.returncode==0:
            resumen=json.loads((carpeta/'resumen.json').read_text(encoding='utf-8'))
            lineas=[json.loads(l) for l in (carpeta/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
            cabecera=lineas[0];frames=[r for r in lineas if r['tipo']=='frame']
            if cabecera['fuente_tipo']!='camara' or len(frames)!=a.max_frames:
                raise ValueError('No se completó la captura física prevista')
            if any(f['estado_detector']!='experimental_validacion_insuficiente' for f in frames):
                raise ValueError('Advertencia de calidad ausente')
            import cv2
            cap=cv2.VideoCapture(str(carpeta/'anotado.mp4'));n=0;primer_frame=None
            while True:
                ok,f=cap.read()
                if not ok:break
                if primer_frame is None:primer_frame=f
                n+=1
            cap.release()
            if n!=len(frames):raise ValueError('MP4 de captura incompleto')
            if primer_frame is not None:cv2.imwrite(str(carpeta/'primer_frame.jpg'),primer_frame)
            resultado.update(estado='completada',captura_real_verificada=True,frames_capturados=len(frames),
                frames_mp4_releidos=n,resolucion=[cabecera['ancho'],cabecera['alto']],
                mediana_inferencia_ms=resumen['latencia_inferencia_mediana_ms'],
                p95_inferencia_ms=resumen['latencia_inferencia_p95_ms'],fps_procesamiento=resumen['fps_procesamiento'],
                politica=resumen['politica'],vistas=resumen['vistas'],salida=carpeta.relative_to(RAIZ).as_posix(),
                sha256_mp4=hashlib.sha256((carpeta/'anotado.mp4').read_bytes()).hexdigest(),
                limite='Prueba funcional de captura e inferencia, sin anotaciones; no mide precisión ni mAP en cámara.')
        else:
            resultado.update(estado='no_se_pudo_completar',captura_real_verificada=False,
                limite='La apertura/lectura falló; el error no identifica por sí solo ausencia de hardware, permisos, dispositivo ocupado o privacidad.')
    except subprocess.TimeoutExpired as e:
        resultado.update(estado='timeout',captura_real_verificada=False,
            stdout=e.stdout.decode('utf-8',errors='replace') if isinstance(e.stdout,bytes) else e.stdout,
            stderr=e.stderr.decode('utf-8',errors='replace') if isinstance(e.stderr,bytes) else e.stderr,
            limite='La prueba se terminó al alcanzar el límite de tiempo; no se declara la cámara validada.')
    except Exception as e:
        resultado.update(estado='error_verificacion',captura_real_verificada=False,error=str(e))
    resultado['duracion_segundos']=time.perf_counter()-inicio
    resultado['sha256_seleccion_despues']=hashlib.sha256(config.read_bytes()).hexdigest()
    if resultado['sha256_seleccion_despues']!=sha_antes:raise ValueError('La selección cambió durante la prueba')
    escribir_json(informe,resultado)
    print(json.dumps({k:v for k,v in resultado.items() if k not in ['stdout','stderr']},ensure_ascii=False))
    return 0 if resultado['captura_real_verificada'] else 1


if __name__=='__main__':raise SystemExit(main())
