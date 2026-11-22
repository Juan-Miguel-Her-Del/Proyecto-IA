"""Certificado explícito de revisión postentrenamiento: no altera sus hashes/checkpoints."""
import ast
import base64
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import huellas_codigo,protocolo


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    root=RAIZ/'resultados/v24_v20_evaluacion';cfg=leer(RAIZ/'configuracion/v20_backend_v23_1_completo_eval24_ep19.json')
    t=leer(RAIZ/f"resultados/{cfg['nombre']}.json");antes=leer(root/'evaluador_usado_entrenamiento.json');actual=huellas_codigo()
    if antes['hashes_codigo']!=t['hashes_codigo'] or antes['sha256']!=t['hashes_codigo']['scripts/evaluar_v20.py']:raise ValueError('Copia original no corresponde al entrenamiento')
    if set(actual)!=set(t['hashes_codigo']) or [p for p in actual if actual[p]!=t['hashes_codigo'][p]]!=['scripts/evaluar_v20.py']:raise ValueError('Cambios fuera de evaluación')
    arbol=ast.parse(base64.b64decode(antes['bytes_base64']).decode('utf-8-sig'));nuevo=ast.parse((RAIZ/'scripts/evaluar_v20.py').read_text(encoding='utf-8'))
    def funciones(a):return {n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a=funciones(arbol);b=funciones(nuevo);nombres=['inferir_sin_gt','contexto_evaluacion','comparar_predicciones_historicas','verificar_referencia_historica','comparar_control_v18']
    if any(a[n]!=b[n] for n in nombres):raise ValueError('Cambió la inferencia o la referencia histórica')
    d=leer(root/'revision_cpu_diagnostico/diagnostico.json')
    if not d['aprobada'] or d['hashes_codigo']!=actual or d['protocolo']!=protocolo(cfg):raise ValueError('Diagnóstico de revisión pendiente')
    publicar(root/'compatibilidad_evaluacion_cpu.json',{'aprobada':True,'version_evaluacion':'v24_cpu_gt_revision_2',
        'hashes_entrenamiento':t['hashes_codigo'],'hashes_evaluacion':actual,'protocolo':protocolo(cfg),
        'cambio_exclusivo':'scripts/evaluar_v20.py','funciones_de_inferencia_inalteradas_por_AST':nombres,
        'agregacion_GT':'CPU; forward GPU V23 y determinismo estricto conservados',
        'pruebas':d['pruebas'],'sha256_diagnostico_revision':sha(root/'revision_cpu_diagnostico/diagnostico.json'),
        'sha256_pruebas_regresion_cpu':sha(RAIZ/'pruebas/test_agregacion_evaluacion_cpu.py'),
        'checkpoints_y_certificados_entrenamiento_no_modificados':True,'no_autoriza_nuevo_entrenamiento':True,
        'hashes_de_entrenamiento_se_conservan':True,'test_evaluado':False})
    print('Compatibilidad de evaluación certificada; entrenamiento y forward inalterados')


if __name__=='__main__':main()
