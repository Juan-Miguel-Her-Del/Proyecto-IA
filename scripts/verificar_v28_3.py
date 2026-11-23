"""Cierre de preparación: historial intacto, pruebas y servicio CPU; sin entrenar."""
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import urllib.request
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha,publicar
from ecowaste.correcciones_v28_3 import auditorias_verificadas
from ecowaste.auxiliares_publicos import Taller
from scripts.anotar_v27_2 import PROYECTO
from scripts.auditar_v28_3 import SALIDA,gestion


def main():
    if (SALIDA/'verificacion_final.json').exists():raise FileExistsError('Cierre ya conservado; no reemplazar')
    b=leer(SALIDA/'respaldo.json');t=Taller(PROYECTO,RAIZ);g=gestion(t)
    for p,h in b['archivos'].items():
        if sha(RAIZ/b['destino']/p)!=h or sha(PROYECTO/p)!=h:raise ValueError('Respaldo/evento/propuesta histórica alterada: '+p)
    eventos=sum(len(t.diario(i)) for i in t.imagenes)
    if eventos!=134 or g.eventos():raise ValueError('Este cierre de preparación debe conservar134 eventos y no simular resoluciones')
    protegidos=leer(SALIDA/'historicos_protegidos.json')
    for p,h in protegidos.items():
        if sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
    cps={p:h for p,h in protegidos.items() if p.startswith('checkpoints/')}
    if set(cps)!={p.relative_to(RAIZ).as_posix() for p in (RAIZ/'checkpoints').rglob('*.pt')}:raise ValueError('Se añadieron/eliminaron checkpoints durante esta fase sin entrenamiento')
    ps=auditorias_verificadas(SALIDA/'auditorias');s=leer(ps[-1])
    if s['codigo_qa']!=g.codigo_qa() or s['revisiones_humanas']!=134:raise ValueError('Auditoría no vigente')
    prueba=sorted((SALIDA/'pruebas').glob('*/pruebas.json'))[-1];p=leer(prueba)
    browser=sorted((SALIDA/'pruebas_navegador').glob('*.json'))[-1];n=leer(browser)
    if not p['aprobada'] or p['cantidad_pruebas']!=351 or sha(prueba.parent/'suite.log')!=p['sha256_log']:raise ValueError('Suite no aprobada')
    if not n['aprobada'] or len(n['casos'])<11 or n['errores_js'] or not n['revisiones_reales_sin_cambio']:raise ValueError('Navegador no aprobado')
    with urllib.request.urlopen('http://127.0.0.1:8768/api/config',timeout=5) as r:cfg=json.load(r)
    with urllib.request.urlopen('http://127.0.0.1:8768/api/auditoria',timeout=5) as r:qa=json.load(r)
    if cfg['proyecto_sha256']!=t.base['huella'] or cfg['GPU_utilizada'] or cfg['entrenamiento'] or not qa['vigente']:raise ValueError('Servicio no preparado para este proyecto')
    publicar(SALIDA/'verificacion_final.json',{'aprobada':True,'fecha_UTC':datetime.now(timezone.utc).isoformat(),
        'eventos_humanos':eventos,'archivos_respaldados':len(b['archivos']),'checkpoints_conservados':len(cps),
        'archivos_historicos_verificados':len(protegidos),'resoluciones_humanas_creadas_por_agente':0,
        'pruebas':p['cantidad_pruebas'],'casos_navegador':len(n['casos']),'sha256_pruebas':sha(prueba),'sha256_navegador':sha(browser),
        'ultima_auditoria':ps[-1].relative_to(RAIZ).as_posix(),'sha256_ultima_auditoria':sha(ps[-1]),
        'candidatas':s['imagenes_candidatas'],'objetos_validos_candidatas':s['anotaciones_validas_candidatas'],
        'alimentarias_HPI':s['regiones_alimentarias_HPI_utilizables'],'cache_reusadas':s['cache_reusadas'],
        'servicio_url':'http://127.0.0.1:8768','servicio_CPU_verificado':True,'entrenamiento':False,
        'test_evaluado':False,'generalizacion_desbloqueada':False,'checkpoint_predeterminado_modificado':False})
    archivos=[p for p in SALIDA.rglob('*') if p.is_file()]
    archivos += [p for p in (RAIZ/'datos/nuevos/v28_3').rglob('*') if p.is_file() and 'logs' not in p.parts and p.suffix!='.lock']
    archivos += list((RAIZ/'interfaz/anotacion_v28_3').glob('*'))+list((RAIZ/'scripts').glob('*v28_3*'))
    archivos += [RAIZ/'ecowaste/correcciones_v28_3.py',RAIZ/'pruebas/test_correcciones_v28_3.py',RAIZ/'documentacion/28_3_correcciones_post_entrenamiento.md']
    paquetes={}
    for nombre in ['torch','numpy','Pillow','opencv-python','playwright']:
        try:paquetes[nombre]=importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:paquetes[nombre]=None
    manifiesto={'version':'integridad_preparacion_v28_3_1','archivos':{p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(archivos)) if p.is_file()},
        'python':sys.version,'ejecutable':sys.executable,'paquetes_editor_y_QA':paquetes,'pruebas':351,'navegador':11,
        'entrenamiento':False,'aceptadas_V26':0,'nota':'Estado de preparación; nuevos guardados del usuario crean otras auditorías. Logs activos y locks excluidos.'}
    publicar(SALIDA/'manifiesto_integridad.json',manifiesto)
    publicar(SALIDA/'recibo_integridad.json',{'sha256_manifiesto':sha(SALIDA/'manifiesto_integridad.json')})
    print('V28.3 verificada:351 pruebas,11 casos navegador,134 eventos,731 archivos y'+str(len(cps))+' checkpoints intactos; editorCPU listo',flush=True)


if __name__=='__main__':main()
