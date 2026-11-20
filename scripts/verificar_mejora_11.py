"""Verificación final de datos, checkpoints, selección e inferencia real de ambos modelos."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.asistente import Asistente
from ecowaste.datasets import objetivo_deteccion, CLASES
from ecowaste.cajas import deshacer_letterbox
from scripts.entrenar_clasificador import escribir_json


def leer(p): return json.loads((RAIZ / p).read_text(encoding='utf-8'))


def exigir(valor, mensaje):
    if not valor: raise ValueError(mensaje)


def ejecutar(argumentos):
    r = subprocess.run([sys.executable, *argumentos], cwd=RAIZ, capture_output=True,
        text=True, encoding='utf-8', env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
    if r.returncode: raise RuntimeError(r.stderr)
    return json.loads(r.stdout)


def main():
    torch.set_num_threads(4)
    valid = leer('datos/reportes/validacion.json')
    for ruta, sha in valid['sha256_manifiestos'].items():
        exigir(hashlib.sha256((RAIZ/ruta).read_bytes()).hexdigest()==sha, f'Manifiesto alterado: {ruta}')
    nombres_c = ['clasificador_piloto_v1', 'clasificador_v2_coseno80', 'clasificador_v3_regularizado']
    nombres_d = ['detector_mapas_v6_auxiliar_ext48', 'detector_centros_v11']
    todos = {n: leer(f'resultados/{n}.json') for n in nombres_c+nombres_d}
    resumen = {}
    for n,r in todos.items():
        ck = torch.load(RAIZ/r['checkpoint_mejor'], map_location='cpu', weights_only=True)
        exigir(r['estado']=='piloto_completado' and not r['test_evaluado'] and not ck['pesos_externos'], n)
        exigir(ck['hashes_manifiestos']==r['hashes_manifiestos'], f'Origen: {n}')
        exigir(ck['epoca']==r['mejor_epoca'], f'Época: {n}')
        clave = 'f1_macro' if n in nombres_c else 'map50'
        valor_ck = ck['mejor_f1'] if n in nombres_c else ck['mejor_map']
        exigir(abs(valor_ck-r['val'][clave])<1e-12, f'Métrica: {n}')
        if n=='detector_centros_v11':
            exigir(all(h['objetos_asignados_train']==2505 for h in r['historial']), 'Asignación incompleta')
        resumen[n] = hashlib.sha256((RAIZ/r['checkpoint_mejor']).read_bytes()).hexdigest()
        del ck
    variantes = {n: leer(f'resultados/{n}.json') for n in ['detector_mapas_v6_mosaico', 'detector_centros_v11_mosaico']}
    for n,r in variantes.items():
        exigir(r['estado']=='reevaluacion_val_completada' and not r['test_evaluado'], n)
        exigir(hashlib.sha256((RAIZ/r['checkpoint']).read_bytes()).hexdigest()==r['sha256_checkpoint'], f'Pesos modificados: {n}')
    config = leer('configuracion/arquitectura.json')
    c = max(nombres_c,key=lambda n:todos[n]['val']['f1_macro'])
    candidatos = {n: todos[n] for n in nombres_d} | variantes
    d = max(candidatos,key=lambda n:candidatos[n]['val']['map50'])
    exigir(config['clasificador']['experimento_seleccionado']==c, 'Clasificador inferior')
    exigir(config['detector']['experimento_seleccionado']==d, 'Detector inferior')
    rcls = json.loads((RAIZ/'datos/procesados/clasificacion/val.jsonl').read_text(encoding='utf-8').splitlines()[0])
    cl = ejecutar(['scripts/clasificar_imagen.py', rcls['ruta'], '--salida', 'resultados/ejemplo_mejora_11_clasificador.json'])
    exigir(Path(cl['checkpoint']).resolve()==(RAIZ/config['clasificador']['checkpoint']).resolve(), 'CLI clasificador desactualizada')
    exigir(cl['f1_macro_validacion_checkpoint']==config['clasificador']['f1_macro_validacion'], 'F1 de otra fuente')
    rdet = json.loads((RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf-8').splitlines()[0])
    det = ejecutar(['scripts/detectar_imagen.py', rdet['ruta'], '--json-salida', 'resultados/ejemplo_mejora_11_detector.json',
                   '--imagen-salida', 'resultados/figuras/ejemplo_mejora_11_detector.jpg'])
    exigir(det['map50_validacion_inferencia_seleccionada']==config['detector']['map50_validacion'], 'AP de otra inferencia')
    exigir(det['estado_detector']=='experimental_validacion_insuficiente', 'Advertencia perdida')
    pred = leer(f'resultados/{d}_predicciones_val.json')[0]
    exigir(pred['id']==rdet['id'], 'Escena diferente')
    keep = torch.tensor(pred['scores']) >= det['umbral_puntuacion']
    boxes = deshacer_letterbox(torch.tensor(pred['boxes'],dtype=torch.float32).reshape(-1,4)[keep], objetivo_deteccion(rdet)['geometria'])
    etiquetas = torch.tensor(pred['labels'])[keep].tolist()
    esperado = torch.tensor(sorted((c,*b.tolist()) for c,b in zip(etiquetas,boxes)),dtype=torch.float32).reshape(-1,5)
    actual = torch.tensor(sorted((CLASES.index(b['material']),*b['bbox_xyxy']) for b in det['detecciones']),dtype=torch.float32).reshape(-1,5)
    exigir(esperado.shape==actual.shape and torch.allclose(esperado,actual,atol=.02,rtol=0),
           'Las cajas CLI no reproducen validación al mismo umbral')
    respuesta = Asistente(RAIZ/'checkpoints/asistente_v1.pt').consultar('¿Cómo separo estos residuos?',det['detecciones'])
    respuesta.update(estado_detector=det['estado_detector'],limite_detector=det['limite'],origen_detecciones=det['origen'])
    escribir_json(RAIZ/'resultados/ejemplo_mejora_11_asistente.json',respuesta)
    pruebas = leer('resultados/pruebas_mejora_11.json')
    exigir(pruebas['correcto'], 'Hay pruebas fallidas')
    informe = {'verificacion_correcta':True,'manifiestos_invariantes':len(valid['sha256_manifiestos']),
        'checkpoints_sha256':resumen,'clasificador_seleccionado':c,'detector_seleccionado':d,
        'pruebas_aprobadas':pruebas['ejecutadas'],'prediccion_primera_clasificacion_val':cl['material_predicho'],
        'clase_real_primera_clasificacion_val':rcls['clase'],'cajas_primera_deteccion_val':len(det['detecciones']),
        'cli_reproduce_cajas_val':True,'asistente_integrado_con_advertencia':True,'test_evaluado':False,'pesos_externos':False}
    escribir_json(RAIZ/'resultados/verificacion_mejora_11.json',informe)
    print(json.dumps({k:v for k,v in informe.items() if k!='checkpoints_sha256'}))


if __name__=='__main__':main()
