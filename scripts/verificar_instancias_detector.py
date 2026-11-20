"""Comprueba procedencia, selección e integración sin evaluar test ni entrenar."""
import hashlib
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.asistente import Asistente


def leer(ruta):
    return json.loads((RAIZ / ruta).read_text(encoding='utf-8'))


def exigir(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


def main():
    torch.set_num_threads(4)
    validacion = leer('datos/reportes/validacion.json')
    for ruta, esperado in validacion['sha256_manifiestos'].items():
        exigir(hashlib.sha256((RAIZ / ruta).read_bytes()).hexdigest() == esperado,
               f'Manifiesto modificado: {ruta}')
    nombres = ['detector_mapas_v6_auxiliar_ext48', 'detector_mapas_v7_ventanas',
               'detector_consultas_v8', 'detector_consultas_v9_centros',
               'detector_consultas_v10_balance_posicion']
    resultados = {n: leer(f'resultados/{n}.json') for n in nombres}
    resumen = {}
    for nombre, r in resultados.items():
        exigir(r['estado'] == 'piloto_completado', f'Piloto pendiente: {nombre}')
        exigir(not r['test_evaluado'] and not r['pesos_externos'], nombre)
        ck = torch.load(RAIZ / r['checkpoint_mejor'], map_location='cpu', weights_only=True)
        exigir(ck['epoca'] == r['mejor_epoca'], f'Época inconsistente: {nombre}')
        exigir(abs(ck['mejor_map'] - r['val']['map50']) < 1e-12, f'AP inconsistente: {nombre}')
        exigir(ck['hashes_manifiestos'] == r['hashes_manifiestos'], f'Procedencia inconsistente: {nombre}')
        for clave, tarea, split in [('train', 'deteccion', 'train'), ('val', 'deteccion', 'val'),
                                   ('clasificacion_auxiliar_train', 'clasificacion', 'train')]:
            exigir(r['hashes_manifiestos'][clave] == validacion['sha256_manifiestos'][
                f'datos/procesados/{tarea}/{split}.jsonl'], f'Datos diferentes: {nombre}/{clave}')
        if nombre.startswith('detector_consultas'):
            exigir(all(e['objetos_asignados_train'] == 2505 for e in r['historial']),
                   f'Asignación incompleta: {nombre}')
        resumen[nombre] = {'map50_val': r['val']['map50'], 'mejor_epoca': ck['epoca'],
                          'sha256_checkpoint': hashlib.sha256((RAIZ / r['checkpoint_mejor']).read_bytes()).hexdigest()}
        del ck
    cfg = leer('configuracion/arquitectura.json')['detector']
    elegido = max(resultados, key=lambda n: resultados[n]['val']['map50'])
    historico = (RAIZ/'documentacion/11_mejora_clasificador_y_centros.md').exists()
    if not historico:
        exigir(cfg['experimento_seleccionado'] == elegido, 'Selección inferior al mejor piloto')
        exigir(cfg['checkpoint'] == resultados[elegido]['checkpoint_mejor'], 'Ruta predeterminada incorrecta')
        exigir(cfg['map50_validacion'] == resultados[elegido]['val']['map50'], 'AP predeterminado incorrecto')
    else:
        exigir(cfg['map50_validacion'] >= resultados[elegido]['val']['map50'], 'Regresión frente al histórico')
    geometria = leer('resultados/geometria_consultas_v10_train.json')
    exigir(geometria['sha256_checkpoint'] == resumen[nombres[-1]]['sha256_checkpoint'], 'Diagnóstico de otro checkpoint')
    entrada = leer('resultados/ejemplo_instancias_val_686.json')
    exigir(entrada['map50_validacion_checkpoint'] == resultados[elegido]['val']['map50'], 'Ejemplo de otro modelo')
    respuesta = Asistente(RAIZ / 'checkpoints/asistente_v1.pt').consultar(
        '¿Cómo separo estos residuos?', entrada['detecciones'])
    respuesta.update(origen_detecciones=entrada['origen'], estado_detector=entrada['estado_detector'],
                     limite_detector=entrada['limite'])
    (RAIZ / 'resultados/ejemplo_instancias_asistente.json').write_text(
        json.dumps(respuesta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    informe = {'verificacion_correcta': True, 'manifiestos_invariantes': len(validacion['sha256_manifiestos']),
               'experimentos': resumen, 'seleccionado': elegido, 'integracion_asistente': 'ejecutada_con_advertencia',
               'verificacion_historica': historico, 'seleccion_actual': cfg['experimento_seleccionado'],
               'test_evaluado': False, 'pesos_externos': False}
    (RAIZ / 'resultados/verificacion_instancias_detector.json').write_text(
        json.dumps(informe, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'correcto': True, 'seleccionado': elegido, 'manifiestos_invariantes': 6,
                      'checkpoints_verificados': len(resumen), 'test_evaluado': False}))


if __name__ == '__main__':
    main()
