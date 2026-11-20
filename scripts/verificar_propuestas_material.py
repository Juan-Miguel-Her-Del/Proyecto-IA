"""Verifica integridad, contabilidad de resultados y conservación de la selección del A/B."""
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from scripts.entrenar_clasificador import escribir_json


def sha(p):return hashlib.sha256((RAIZ/p).read_bytes()).hexdigest()
def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))
def exigir(c,m):
    if not c:raise ValueError(m)


def main():
    r=leer('resultados/mapas_v6_ab_propuestas_material.json')
    exigir(r['estado']=='comparacion_val_completada','Comparación incompleta')
    exigir(r['estado_modelo_inalterado'] and r['parametros_congelados'] and r['logits_compartidos_entre_politicas'],'Protocolo alterado')
    exigir(not r['entrenamiento_realizado'] and not r['test_evaluado'] and not r['seleccion_modificada'],'Ámbito alterado')
    exigir(sha(r['checkpoint'])==r['sha256_checkpoint_antes']==r['sha256_checkpoint_despues'],'Checkpoint modificado')
    exigir(sha('configuracion/arquitectura.json')==r['sha256_seleccion_antes']==r['sha256_seleccion_despues'],'Selección modificada')
    for p,h in r['sha256_manifiestos_invariantes'].items():exigir(sha(p)==h,f'Manifiesto: {p}')
    for p,h in r['sha256_codigo'].items():exigir(sha(p)==h,f'Código del experimento modificado: {p}')
    previo=leer('resultados/verificacion_mejora_11.json')
    clas=leer('configuracion/arquitectura.json')['clasificador']
    exigir(sha(clas['checkpoint'])==previo['checkpoints_sha256'][clas['experimento_seleccionado']],'Clasificador modificado')
    for vistas,base in r['regresion_bases'].items():
        exigir(base['imagenes_con_cajas_etiquetas_y_scores_coincidentes']==r['imagenes_val'],'Base no reproducida')
        exigir(base['maximo_delta_scores_con_cajas_coincidentes']==0,'Base numéricamente distinta')
        exigir(sha(base['referencia'])==base['sha256_referencia'],'Referencia alterada')
    for vistas,cc in r['admisiones_celdas'].items():
        for c,t in cc.items():
            if c!='celdas_validas':
                exigir(t['admisiones_sin_argmax']==t['admisiones_argmax']+t['excluidas_por_fondo_ganador']+t['excluidas_por_otro_material_ganador'],'Conteo de celdas inconsistente')
    total_cajas=0;bordes_redondeados=0;maximo_exceso=0.
    for k,c in r['resultados'].items():
        exigir(sha(c['predicciones_val'])==c['sha256_predicciones_val'],f'Predicciones alteradas: {k}')
        pp=leer(c['predicciones_val'])
        exigir(len(pp)==181 and len({p['id'] for p in pp})==181,'Escenas incompletas')
        for p in pp:
            exigir(len(p['boxes'])==len(p['scores'])==len(p['labels'])<=100,'Forma de predicciones')
            for b,s,l in zip(p['boxes'],p['scores'],p['labels']):
                # La proyección float32 de recortes puede exceder uno en un ULP.
                exigir(-1e-6<=b[0]<b[2]<=1+1e-6 and -1e-6<=b[1]<b[3]<=1+1e-6
                       and .05<=s<=1 and 0<=l<6,'Propuesta inválida')
                exceso=max(0.,-min(b),max(b)-1)
                if exceso>0:bordes_redondeados+=1;maximo_exceso=max(maximo_exceso,exceso)
        for umbral,mm in c['por_filtro_caja'].items():
            s=mm['resumen'];m=mm['metricas']
            n=sum(sum(score>=float(umbral) for score in p['scores']) for p in pp)
            exigir(s['cajas_totales']==s['tp50']+s['fp50']+s['predicciones_ignoradas50']==n,'Contabilidad de cajas')
            exigir(s['tp50']+s['fn50']==m['objetos']==361,'Contabilidad de objetos')
            exigir(abs(s['recall50']-s['tp50']/361)<1e-12,'Recall inconsistente')
        total_cajas+=sum(len(p['boxes']) for p in pp)
    pruebas=leer('resultados/pruebas_propuestas_material.json')
    exigir(pruebas['correcto'],'Pruebas fallidas')
    for p in ['documentacion/12_propuestas_por_material_sin_argmax.md','resultados/figuras/propuestas_material_ab.png',
              'resultados/figuras/propuestas_material_ejemplo.png']:exigir((RAIZ/p).is_file(),f'Artefacto ausente: {p}')
    informe={'correcto':True,'condiciones_verificadas':4,'imagenes_por_condicion':181,
        'cajas_registradas_en_las_cuatro_condiciones':total_cajas,'pruebas_aprobadas':pruebas['ejecutadas'],
        'cajas_con_redondeo_float32_en_borde':bordes_redondeados,'maximo_exceso_normalizado':maximo_exceso,
        'checkpoint_y_clasificador_inalterados':True,'seis_manifiestos_invariantes':True,
        'bases_reproducidas_exactamente':True,'seleccion_modificada':False,'entrenamiento_realizado':False,'test_evaluado':False}
    escribir_json(RAIZ/'resultados/verificacion_propuestas_material.json',informe)
    print(json.dumps(informe))


if __name__=='__main__':main()
