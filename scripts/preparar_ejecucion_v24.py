"""Revalida por alcance los controles V23 sin reescribir ni repetir sus 125 pasos."""
import argparse
import ast
import base64
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import leer_cfg,protocolo,validar,huellas_codigo

ROOT=RAIZ/'resultados/v24_v20_evaluacion'
CONFIG=RAIZ/'configuracion/v20_backend_v23_1_completo_eval24_ep19.json'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def configurar():
    c=leer(RAIZ/'configuracion/v20_backend_v23_1_completo_ep19.json')
    c.update(nombre='v20_backend_v23_1_completo_eval24_ep19',integracion_evaluacion='v24_backend_coherente_historico_separado_v1',
        diagnostico='resultados/v24_v20_evaluacion/preparacion/diagnostico.json',
        prueba_corta='resultados/v24_v20_evaluacion/controles/corto_revalidado.json',
        control_backend='resultados/v24_v20_evaluacion/controles/largo_revalidado.json',
        referencia_historica='resultados/v24_v20_evaluacion/referencia_historica.json',
        control_etapa_20='resultados/v24_v20_evaluacion/etapa20/verificacion.json',entrenamiento_completo_habilitado=False)
    validar(c);publicar(CONFIG,c);print(CONFIG)


def certificar():
    c=leer(CONFIG);d=leer(RAIZ/c['diagnostico']);h=huellas_codigo()
    if not d['aprobada'] or d['hashes_codigo']!=h or d['protocolo']!=protocolo(c):raise ValueError('Diagnóstico pendiente')
    antes=leer(ROOT/'estado_previo.json');copias=leer(ROOT/'codigo_previo.json');permitidos=set(antes['cambios_codigo_autorizados'])
    for p,x in antes['archivos'].items():
        if p not in permitidos and sha(RAIZ/p)!=x:raise ValueError('Histórico alterado: '+p)
    def normalizar(s):
        arbol=ast.parse(s);arbol.body=[n for n in arbol.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODIGOS' for t in n.targets))]
        return ast.dump(arbol,include_attributes=False)
    anterior=base64.b64decode(copias['scripts/preparar_v20.py']['bytes_base64']).decode('utf-8-sig')
    if normalizar(anterior)!=normalizar((RAIZ/'scripts/preparar_v20.py').read_text(encoding='utf-8-sig')):raise ValueError('Cambios fuera de la lista de integridad')
    evidencia=leer(RAIZ/'resultados/v23_backend_v20/decision.json');viejas=evidencia['hashes_codigo']
    cambios=[p for p,x in viejas.items() if h.get(p)!=x]
    nuevas=sorted(set(h)-set(viejas))
    if set(cambios)!=permitidos or nuevas!=['pruebas/test_evaluacion_v20.py']:raise ValueError('Cambios de entrenamiento no cubiertos')
    largo=leer(RAIZ/'resultados/v23_backend_v20/largo_completo.json');corto=leer(RAIZ/'resultados/v23_backend_v20/corto_completo.json')
    if not evidencia['aprobada'] or not largo['aprobada'] or not corto['aprobada'] or evidencia['backend_numerico']!=c['backend_numerico']:raise ValueError('Evidencia V23 no transferible')
    informe={'aprobada':True,'cambios_codigo':cambios,'pruebas_nuevas':nuevas,'funciones_preparador_sin_cambios':True,
        'entrenador_modelo_perdida_backend_y_reanudacion_identicos_por_sha256':True,
        'evidencia_125_no_repetida_ni_modificada':True,'hashes_codigo_previos':viejas,'hashes_codigo_actuales':h,
        'sha256_manifiesto_V23':sha(RAIZ/'resultados/v23_backend_v20/manifiesto_integridad.json'),
        'motivo':'Solo cambia la integración de evaluación y la lista de pruebas. La computación de los 125 pasos conserva exactamente sus dependencias.',
        'test_evaluado':False}
    publicar(ROOT/'revalidacion_por_alcance.json',informe)
    base={'aprobada':True,'hashes_codigo':h,'sha256_diagnostico':sha(RAIZ/c['diagnostico']),
        'backend_numerico':c['backend_numerico'],'protocolo':protocolo(c),
        'revalidacion_por_alcance':'resultados/v24_v20_evaluacion/revalidacion_por_alcance.json',
        'sha256_revalidacion':sha(ROOT/'revalidacion_por_alcance.json'),'pasos_125_repetidos':False}
    for campo,fuente in [('prueba_corta','corto_completo.json'),('control_backend','largo_completo.json')]:
        p=RAIZ/'resultados/v23_backend_v20'/fuente
        publicar(RAIZ/c[campo],{**base,'fuente_original':p.relative_to(RAIZ).as_posix(),'sha256_fuente_original':sha(p)})
    print('Certificados revalidados por alcance; evidencia de 125 pasos intacta')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--certificar',action='store_true');a=p.parse_args()
    certificar() if a.certificar else configurar()
