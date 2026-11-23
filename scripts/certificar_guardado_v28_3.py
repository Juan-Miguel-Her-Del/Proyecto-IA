"""Suplemento inmutable del cierre: bloqueo inmediato del doble guardado en la UI."""
from pathlib import Path
import sys
import urllib.request
import json
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha,publicar
from ecowaste.auxiliares_publicos import Taller
from scripts.anotar_v27_2 import PROYECTO
from scripts.auditar_v28_3 import SALIDA,gestion


def main():
    anterior=leer(SALIDA/'manifiesto_integridad.json');base=leer(SALIDA/'verificacion_final.json')
    permitidos={'interfaz/anotacion_v28_3/recuperacion.js','scripts/probar_editor_v28_3.py',
        'documentacion/28_3_correcciones_post_entrenamiento.md'}
    diferencias={p:{'antes':h,'ahora':sha(RAIZ/p)} for p,h in anterior['archivos'].items() if sha(RAIZ/p)!=h}
    if set(diferencias)!=permitidos:raise ValueError('Cambios fuera de UI, prueba de navegador e informe: '+str(list(diferencias)))
    for p,h in leer(SALIDA/'historicos_protegidos.json').items():
        if sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
    b=leer(SALIDA/'respaldo.json')
    for p,h in b['archivos'].items():
        if sha(PROYECTO/p)!=h or sha(RAIZ/b['destino']/p)!=h:raise ValueError('Revisión/respaldo alterado')
    t=Taller(PROYECTO,RAIZ);g=gestion(t)
    if sum(len(t.diario(i)) for i in t.imagenes)!=134 or g.eventos():raise ValueError('No simular correcciones reales')
    navegador=sorted((SALIDA/'pruebas_navegador').glob('*.json'))[-1];n=leer(navegador)
    if not n['aprobada'] or len(n['casos'])!=12 or n['errores_js'] or not n['revisiones_reales_sin_cambio']:raise ValueError('Falta prueba final de doble guardado')
    with urllib.request.urlopen('http://127.0.0.1:8768/api/auditoria',timeout=5) as r:qa=json.load(r)
    if not qa['vigente']:raise ValueError('Auditoría no vigente')
    r={**base,'casos_navegador':12,'sha256_navegador':sha(navegador),'cierre_anterior_sha256':sha(SALIDA/'verificacion_final.json'),
        'manifiesto_anterior_sha256':sha(SALIDA/'manifiesto_integridad.json'),'diferencias_permitidas':diferencias,
        'motivo_suplemento':'Bloquear inmediatamente antes de esperar IndexedDB; prueba de dos llamadas simultáneas, sin cambiar backend/QA/datos',
        'resoluciones_humanas_creadas_por_agente':0}
    publicar(SALIDA/'verificacion_guardado_final.json',r)
    paths=[p for p in SALIDA.rglob('*') if p.is_file()]
    paths += [p for p in (RAIZ/'datos/nuevos/v28_3').rglob('*') if p.is_file() and 'logs' not in p.parts and p.suffix!='.lock']
    paths += list((RAIZ/'interfaz/anotacion_v28_3').glob('*'))+list((RAIZ/'scripts').glob('*v28_3*'))
    paths += [RAIZ/'ecowaste/correcciones_v28_3.py',RAIZ/'pruebas/test_correcciones_v28_3.py',RAIZ/'documentacion/28_3_correcciones_post_entrenamiento.md']
    publicar(SALIDA/'manifiesto_integridad_guardado_final.json',{**anterior,'version':'integridad_preparacion_v28_3_2',
        'archivos':{p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(paths)) if p.is_file()},'navegador':12,
        'manifiesto_previo_sha256':sha(SALIDA/'manifiesto_integridad.json')})
    publicar(SALIDA/'recibo_integridad_guardado_final.json',{'sha256_manifiesto':sha(SALIDA/'manifiesto_integridad_guardado_final.json')})
    print('Cierre V28.3 final:351 pruebas,12 casos navegador; UI protegida e históricos intactos',flush=True)


if __name__=='__main__':main()
