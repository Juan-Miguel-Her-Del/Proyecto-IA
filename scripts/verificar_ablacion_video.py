"""Verificación final de estados, pasos, trazas, métricas, predeterminado y video."""
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.ablacion_fondo import huella,metricas_semanticas,resumir_deteccion
from scripts.entrenar_clasificador import escribir_json


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((RAIZ/p).read_bytes()).hexdigest()
def exigir(c,m):
    if not c:raise ValueError(m)


def main():
    torch.set_num_threads(4)
    r=leer('resultados/ablacion_fondo_ep19.json')
    exigir(r['estado']=='ablacion_entrenamiento_completada','Entrenamiento incompleto')
    exigir(sha(r['checkpoint_origen'])==r['sha256_origen']==r['sha256_origen_despues'],'Origen modificado')
    exigir(sha('configuracion/arquitectura.json')==r['sha256_seleccion_antes']==r['sha256_seleccion_despues'],'Selección modificada')
    seleccion=leer('configuracion/arquitectura.json')
    exigir(seleccion['detector']['experimento_seleccionado']=='detector_mapas_v6_mosaico','Cambió el modo predeterminado')
    exigir(seleccion['detector']['modo_inferencia']=='mosaico_cinco_vistas','Se cambió el número de vistas')
    for p,h in r['sha256_manifiestos_invariantes'].items():exigir(sha(p)==h,f'Manifiesto modificado: {p}')
    fuente=torch.load(RAIZ/r['checkpoint_origen'],map_location='cpu',weights_only=True)
    arquitectura_origen=dict(fuente['configuracion'])
    arquitectura_origen.setdefault('ventanas_desplazadas',False)
    h_fuente={'pesos':huella(fuente['pesos']),'adamw':huella(fuente['optimizador']),'scaler':huella(fuente['scaler']),
        'aleatorios':huella({k:fuente[k] for k in ['rng_torch','rng_cuda','rng_loader','rng_auxiliar']})}
    exigir(h_fuente==r['huellas_estado_origen'],'Huellas de origen incorrectas')
    estados_finales={};checkpoints={};filas_gt=torch.tensor(r['semantica_train_origen']['matriz_confusion']).sum(1)
    for rama,coef in [('A',3.),('B',1.)]:
        b=r['ramas'][rama]
        exigir(b['huellas_estado_inicial']==h_fuente,f'Estado inicial diferente: {rama}')
        exigir(b['actualizaciones']==1452 and len(b['historial'])==12,'Presupuesto desigual')
        exigir(b['paso_adamw_inicial']==2299 and b['paso_adamw_final']==3751,'Paso AdamW incorrecto')
        exigir(all(e['learning_rate']==.0005 for e in b['historial']),'Warmup o LR cambiado')
        exigir(all(e['muestras_train']==964 and e['muestras_aux']==1936 for e in b['historial']),'Muestras diferentes')
        for tipo in ['final','mejor']:
            sem=b[f'semantica_train_{tipo}'];recalculada=metricas_semanticas(sem['matriz_confusion'])
            exigir(recalculada==sem,'Métricas semánticas incoherentes')
            exigir(torch.equal(torch.tensor(sem['matriz_confusion']).sum(1),filas_gt),'Las etiquetas train cambiaron')
            exigir(resumir_deteccion(b[f'val_{tipo}_una_vista'])==b[f'resumen_{tipo}'],'Resumen val inconsistente')
            path=b[f'checkpoint_{tipo}'];ck=torch.load(RAIZ/path,map_location='cpu',weights_only=True)
            exigir(ck['hashes_manifiestos']==fuente['hashes_manifiestos'],'Datos de checkpoint distintos')
            exigir(ck['configuracion']==arquitectura_origen,'Arquitectura distinta')
            exigir(ck['pesos'].keys()==fuente['pesos'].keys()
                and all(ck['pesos'][k].shape==v.shape for k,v in fuente['pesos'].items()),'Estructura de pesos distinta')
            exigir(ck['config_entrenamiento']['coeficiente_fondo']==coef,'Coeficiente equivocado')
            esperado=31 if tipo=='final' else b['mejor_epoca_durante_actualizaciones']
            exigir(ck['epoca']==esperado and ck['actualizaciones_desde_origen']==(esperado-19)*121,'Epoca/pasos desalineados')
            exigir({int(s['step']) for s in ck['optimizador']['state'].values()}=={2299+(esperado-19)*121},'Pasos reales incorrectos')
            exigir(all(g['lr']==.0005 for g in ck['optimizador']['param_groups']),'LR del checkpoint cambiado')
            cinco=leer(f'resultados/ablacion_fondo_ep19_{rama}_{tipo}_cinco_vistas.json')
            exigir(cinco['sha256_checkpoint']==sha(path),'Evaluación cinco vistas de otro checkpoint')
            exigir(cinco['val']['objetos']==361 and not cinco['test_evaluado'],'Validación incompleta o test usado')
            checkpoints[path]=sha(path)
            if tipo=='final':estados_finales[rama]={k:ck[k] for k in ['rng_torch','rng_cuda','rng_loader','rng_auxiliar']}
            del ck
        incl=torch.load(RAIZ/b['checkpoint_mejor_incluyendo_origen'],map_location='cpu',weights_only=True)
        exigir(incl['epoca']==b['mejor_epoca_incluyendo_origen'],'Mejor incluyendo origen incoherente')
        if rama=='A':exigir(huella(incl['pesos'])==h_fuente['pesos'],'A debía conservar mejor origen')
        del incl
    exigir(r['ramas']['A']['huella_orden_y_aumentos']==r['ramas']['B']['huella_orden_y_aumentos'],'Trazas distintas')
    exigir(all(a['huella_orden_y_aumentos']==b['huella_orden_y_aumentos'] for a,b in zip(r['ramas']['A']['historial'],r['ramas']['B']['historial'])),'Trazas por época distintas')
    exigir(huella(estados_finales['A'])==huella(estados_finales['B']),'Consumo aleatorio final distinto')
    anterior=leer('resultados/verificacion_mejora_11.json')
    exigir(sha(seleccion['clasificador']['checkpoint'])==anterior['checkpoints_sha256'][seleccion['clasificador']['experimento_seleccionado']],'Clasificador modificado')
    video=leer('resultados/verificacion_video_real.json');imagen=leer('resultados/verificacion_video_imagen.json')
    exigir(video['correcto'] and imagen['correcto'],'Falla integración video')
    exigir(video['frames_inferidos']==video['frames_mp4_exportados']==8,'Frames incompletos')
    exigir(not video['camara_real_activada'] and not video['video_independiente'],'Alcance de la prueba mal declarado')
    exigir(sha(video['entrada'])==video['sha256_entrada'],'Entrada MP4 alterada')
    exigir(sha(video['salida']+'/anotado.mp4')==video['sha256_salida'],'Salida MP4 alterada')
    exigir(video['sha256_seleccion_inalterada']==r['sha256_seleccion_antes'],'Video usó otra selección')
    pruebas=leer('resultados/pruebas_ablacion_video.json');exigir(pruebas['correcto'],'Pruebas fallidas')
    for p in ['documentacion/13_ablacion_del_fondo.md','documentacion/14_video_y_webcam_experimental.md',
        'resultados/figuras/ablacion_fondo_aprendizaje.png','resultados/figuras/ablacion_fondo_confusiones.png',
        'resultados/figuras/ablacion_fondo_cinco_vistas.png']:exigir((RAIZ/p).is_file(),f'Artefacto ausente: {p}')
    informe={'correcto':True,'estado_inicial_completo_igual':True,'orden_y_aumentos_iguales':True,
        'consumo_rng_final_igual':True,'actualizaciones_por_rama':1452,'paso_adamw_final':3751,
        'lr_constante':.0005,'warmup_reiniciado':False,'checkpoint_origen_inalterado':True,
        'predeterminado_inalterado':True,'clasificador_inalterado':True,'video_mp4_verificado':True,
        'webcam_implementada_verificada_con_mock':True,'camara_fisica_verificada':False,
        'pruebas_aprobadas':pruebas['ejecutadas'],'sha256_checkpoints_ramas':checkpoints,'test_evaluado':False}
    escribir_json(RAIZ/'resultados/verificacion_ablacion_video.json',informe)
    print(json.dumps({k:v for k,v in informe.items() if k!='sha256_checkpoints_ramas'}))


if __name__=='__main__':main()
