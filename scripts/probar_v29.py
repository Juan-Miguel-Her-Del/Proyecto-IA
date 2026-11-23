"""Conservar suite histórica completa y ejecutar codec COCO en entorno CPU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
from pathlib import Path
import re,subprocess,sys,uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar_bytes,publicar,sha
from ecowaste.experimento_v29 import codigo,SALIDA


def main():
    out=SALIDA/'pruebas'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    comandos={'suite':[sys.executable,'-m','unittest','discover','-s','pruebas'],
        'codec_cpu':[str(Path.home()/'.venvs/EcoWasteAnotacion/Scripts/python.exe'),'-m','unittest','pruebas.test_datos_y_entrenamiento_v29','-v']}
    resultados={}
    # La suite histórica prueba también el backend mínimo, cuyo contrato requiere
    # CUBLAS_WORKSPACE_CONFIG ausente. Aislar su proceso, sin cambiar el backend
    # completo exigido por el entrenador V29 ni modificar esas pruebas antiguas.
    entorno=os.environ.copy();entorno.pop('CUBLAS_WORKSPACE_CONFIG',None)
    for nombre,cmd in comandos.items():
        r=subprocess.run(cmd,cwd=RAIZ,env=entorno,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace')
        p=out/(nombre+'.log');publicar_bytes(p,r.stdout.encode('utf8'));m=re.search(r'Ran (\d+) tests',r.stdout)
        resultados[nombre]={'codigo_retorno':r.returncode,'pruebas':int(m[1]) if m else 0,'sha256_log':sha(p)}
        print(nombre,resultados[nombre],flush=True)
    ok=resultados['suite']['codigo_retorno']==0 and resultados['suite']['pruebas']>=426 and resultados['codec_cpu']['codigo_retorno']==0 and resultados['codec_cpu']['pruebas']>=17
    publicar(out/'pruebas.json',{'aprobada':ok,'resultados':resultados,'codigo':codigo(),'anteriores':409})
    if not ok:raise RuntimeError('Pruebas fallidas; entrenamiento bloqueado. Revisar '+str(out))


if __name__=='__main__':main()
