"""Comprueba la auditoría CPU, integridad de modelos/datos y evidencia física sin abrir cámara."""
import csv
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import cv2
from ecowaste.datasets import CLASES
from ecowaste.auditoria_deteccion import ESTADOS,TAMANOS
from scripts.entrenar_clasificador import escribir_json


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((RAIZ/p).read_bytes()).hexdigest()
def exigir(c,m):
    if not c:raise ValueError(m)


def main():
    r=leer('resultados/auditoria_clase_tamano.json');cfg=leer('configuracion/arquitectura.json')
    exigir(r['estado']=='auditoria_completada' and r['inferencias_realizadas']==r['actualizaciones_entrenamiento']==0,'Ámbito de auditoría incorrecto')
    exigir(not r['gpu_utilizada'] and r['dispositivo']=='cpu' and not r['test_evaluado'],'Se usó GPU o test en la auditoría')
    exigir(sha('configuracion/arquitectura.json')==r['sha256_seleccion_antes']==r['sha256_seleccion_despues'],'Selección modificada')
    for s,h in r['sha256_pesos'].items():exigir(sha(cfg[s]['checkpoint'])==h,f'Pesos cambiados: {s}')
    for p,h in r['sha256_manifiestos_invariantes'].items():exigir(sha(p)==h,f'Manifiesto cambiado: {p}')
    for p,h in r['fuentes_predicciones_sha256'].items():exigir(sha(p)==h,f'Predicciones cambiadas: {p}')
    exigir(sha(r['fuente_metricas'])==r['sha256_fuente_metricas'],'Métricas fuente cambiadas')
    filas=0
    for key,c in r['condiciones'].items():
        os=c['objetos'];s=c['diagnostico_gt']['total'];m=c['resumen_deteccion_existente']
        exigir(len(os)==s['objetos']==361 and sum(s[e] for e in ESTADOS)==361,'Partición GT incompleta')
        exigir(s['correcto']==m['tp50'] and s['fn']==m['fn50'] and len(c['falsos_positivos'])==m['fp50'],'No se reprodujo el evaluador')
        exigir(sum(c['tipos_fp'].values())==m['fp50'],'FP incompletos')
        exigir(len({(o['id_imagen'],o['id_anotacion']) for o in os})==361,'Se duplicaron GT')
        exigir(sum(x['objetos'] for x in c['diagnostico_gt']['por_tamano'].values())==361,'Tamaños incompletos')
        for clase in CLASES:
            cc=c['diagnostico_gt']['por_clase'][clase]
            exigir(cc['objetos']==sum(o['material_real']==clase for o in os),'Clase inconsistente')
            exigir(sum(c['diagnostico_gt']['por_clase_y_tamano'][clase][t]['objetos'] for t in TAMANOS)==cc['objetos'],'Cruce clase/tamaño incompleto')
        for o in os:
            if o['estado']=='correcto':
                exigir(o['iou_testigo']>=.5 and o['clase_real']==o['material_testigo'],'TP mal asignado')
            elif o['estado']=='material_equivocado':
                exigir(o['iou_testigo']>=.5 and o['clase_real']!=o['material_testigo'] and o['cajas_razonables_material_correcto']==0,'Error de material mal atribuido')
            elif o['estado']=='geometria_incorrecta':
                exigir(o['cobertura_gt_testigo']>=.5 or o['subtipo']=='caja_compartida_sin_asignacion_univoca','Geometría sin cobertura')
            elif o['estado']=='sin_cobertura_suficiente':exigir(o['cobertura_max_gt']<.5,'Ausencia de cobertura mal atribuida')
            else:raise ValueError('Categoría desconocida')
        filas+=len(os)
    exigir(sha(r['csv_objetos'])==r['sha256_csv_objetos'],'CSV por objeto alterado')
    with (RAIZ/r['csv_objetos']).open(encoding='utf-8-sig',newline='') as f:exigir(len(list(csv.DictReader(f)))==filas,'CSV incompleto')
    with (RAIZ/'datos/reportes/revision_clases_sin_ap.csv').open(encoding='utf-8-sig',newline='') as f:
        cola=list(csv.DictReader(f));exigir(len(cola)==48 and all(x['material_real'] in ['carton','vidrio','organico'] for x in cola),'Cola de revisión incompleta')
    exigir(r['cobertura_datos']['train']['organico']['instancias']==5 and r['cobertura_datos']['val']['organico']['instancias']==1,'Conteo orgánico alterado')
    cam=leer('resultados/prueba_camara_fisica_02_fuera_aislamiento.json')
    exigir(cam['captura_real_verificada'] and not cam['mock_utilizado'] and cam['codigo_salida']==0,'Cámara física no verificada')
    exigir(cam['sha256_seleccion_antes']==cam['sha256_seleccion_despues']==r['sha256_seleccion_antes'],'Cámara cambió selección')
    carpeta=cam['salida'];exigir(sha(carpeta+'/anotado.mp4')==cam['sha256_mp4'],'Video de cámara alterado')
    registros=[json.loads(l) for l in (RAIZ/carpeta/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    cab=registros[0];frames=[x for x in registros if x['tipo']=='frame']
    exigir(cab['fuente_tipo']=='camara' and cab['fuente']=='0' and cab['vistas']==5 and cab['politica']=='argmax7','Fuente/motor equivocados')
    exigir(cab['sha256_checkpoint']==r['sha256_pesos']['detector'],'Cámara usó otro checkpoint')
    exigir([f['indice_frame'] for f in frames]==list(range(12)),'Frames incompletos')
    exigir(all(f['estado_detector']=='experimental_validacion_insuficiente' and f['origen_tiempo']=='reloj_captura' for f in frames),'Advertencias o tiempos perdidos')
    cap=cv2.VideoCapture(str(RAIZ/carpeta/'anotado.mp4'));n=0
    while True:
        ok,f=cap.read()
        if not ok:break
        exigir(f.shape[:2]==(480,640),'Resolución de exportación incorrecta');n+=1
    cap.release();exigir(n==cam['frames_capturados']==cam['frames_mp4_releidos']==12,'MP4 físico incompleto')
    dispositivos=leer('resultados/dispositivos_camara_fisica_fuera_aislamiento.json')
    exigir(dispositivos['consulta_correcta'] and any(d['FriendlyName']=='Integrated Camera' and d['Status']=='OK' for d in dispositivos['dispositivos']),'Dispositivo físico no enumerado')
    pruebas=leer('resultados/pruebas_auditoria_camara.json');exigir(pruebas['correcto'],'Pruebas fallidas')
    for p in ['documentacion/15_auditoria_clase_y_tamano.md','documentacion/16_prueba_camara_fisica.md',
        'resultados/figuras/auditoria_fallos_clase.png','resultados/figuras/auditoria_tamanos.png',
        'resultados/figuras/auditoria_materiales.png','resultados/figuras/auditoria_ejemplos.png']:
        exigir((RAIZ/p).is_file(),f'Artefacto ausente: {p}')
    informe={'correcto':True,'condiciones_auditoria':4,'gt_por_condicion':361,'filas_csv_verificadas':filas,
        'registros_revision':48,'auditoria_inferencias_nuevas':0,'auditoria_gpu':False,'entrenamientos':0,
        'camara_fisica_verificada':True,'frames_fisicos':12,'resolucion':[640,480],
        'modelos_y_seleccion_inalterados':True,'manifiestos_invariantes':True,
        'pruebas_aprobadas':pruebas['ejecutadas'],'test_evaluado':False,
        'limite_camara':'Captura e inferencia breve; no evaluación de precisión ni prueba sostenida.'}
    escribir_json(RAIZ/'resultados/verificacion_auditoria_camara.json',informe);print(json.dumps(informe))


if __name__=='__main__':main()
