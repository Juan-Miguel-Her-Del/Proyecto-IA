"""Manifiesto verificable de entrega; no autoriza ni ejecuta entrenamiento."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from scripts.supervision_regional_materiales import CONFIG,sha,hashes_codigo,protocolo
from scripts.entrenar_clasificador import escribir_json


def main():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));base=RAIZ/'resultados/supervision_regional_preparacion'
    salida=base/'entrega.json'
    if salida.exists():raise FileExistsError('Entrega existente')
    diagnostico=RAIZ/cfg['diagnostico'];d=json.loads(diagnostico.read_text(encoding='utf-8'))
    verificacion=base/'regresion_val_revision03.json';v=json.loads(verificacion.read_text(encoding='utf-8'))
    if not d['aprobada'] or not v['aprobada'] or cfg['entrenamiento_completo_habilitado']:raise ValueError('Preparación incompleta')
    if d['hashes_codigo']!=hashes_codigo() or d['protocolo']!=protocolo(cfg) or v['sha256_diagnostico']!=sha(diagnostico):
        raise ValueError('Evidencia desactualizada')
    anterior=json.loads((base/'estado_previo.json').read_text(encoding='utf-8'))
    if any(sha(RAIZ/p)!=h for p,h in anterior['sha256_protegidos'].items()):raise ValueError('Cambios en artefactos protegidos')
    if list((RAIZ/'checkpoints').glob('supervision_regional*')):raise ValueError('Checkpoints completos inesperados')
    doc=RAIZ/'documentacion/17_supervision_regional_materiales.md'
    for destino in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        enlace=(doc.parent/destino).resolve()
        if enlace!=salida.resolve() and not enlace.exists():raise ValueError('Enlace local roto: '+destino)
    archivos=[CONFIG,doc,RAIZ/'README.md',*base.iterdir(),
        *[RAIZ/p for p in hashes_codigo()],RAIZ/'scripts/verificar_supervision_regional.py',
        RAIZ/'scripts/graficar_supervision_regional.py',Path(__file__)]
    escribir_json(salida,{'estado':'implementacion_y_diagnostico_completados_ab_pendiente',
        'fecha_utc':datetime.now(timezone.utc).isoformat(),'pruebas':d['pruebas'],
        'diagnostico_vigente':diagnostico.relative_to(RAIZ).as_posix(),
        'regresion_val':verificacion.relative_to(RAIZ).as_posix(),
        'predicciones_val_exactas':all(x['predicciones_exactas_archivo_anterior'] for x in v['val_congelada'].values()),
        'presupuesto_futuro_por_rama':1452,'actualizaciones_ab_ejecutadas':0,
        'entrenamiento_completo_habilitado':False,'pesos_diagnostico_descartados':True,
        'test_evaluado':False,'cambio_predeterminado':False,
        'instancias_elegibles':d['estadistica_regiones']['instancias_elegibles'],
        'sha256_entregables':{p.relative_to(RAIZ).as_posix():sha(p) for p in archivos if p.is_file()},
        'entorno':{'torch':str(torch.__version__),'cuda_torch':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'matmul_precision':torch.get_float32_matmul_precision(),
            'algoritmos_deterministas':torch.are_deterministic_algorithms_enabled(),
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark}})
    print('Preparación cerrada; A/B no iniciado:',salida)


if __name__=='__main__':main()
