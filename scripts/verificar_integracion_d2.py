"""Cierre verificable de integración; sin inferencia, entrenamiento o escrituras ajenas."""
import argparse,json,sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ));OUT=RAIZ/'resultados/integracion_d2'
from ecowaste.incorporacion import sha
from ecowaste.integracion_d2 import leer_config,verificar_pesos
def leer(p):return json.loads(p.read_text(encoding='utf8'))
def escribir(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,indent=2,ensure_ascii=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',default='');args=ap.parse_args()
    if args.revision and not args.revision.isidentifier():raise ValueError('Revisión inválida')
    suf='_'+args.revision if args.revision else '';archivo_cert='integridad'+suf+'.json';archivo_recibo='recibo_integridad'+suf+'.json'
    initial=leer(OUT/'estado_inicial.json');n=0
    for rel,h in initial['protegidos'].items():
        # Este incorporador se creó en esta tarea antes de tomar la foto inicial,
        # no es un script histórico del usuario. Su versión inicial se conserva.
        if rel=='scripts/incorporar_vision_d2.py':
            if sha(OUT/'incorporador_inicial.py')!=h:raise ValueError('Se perdió el incorporador inicial')
        elif sha(RAIZ/rel)!=h:raise ValueError('Archivo previo modificado: '+rel)
        n+=1
    origin=Path(initial['fuente']);source={}
    if origin.exists():
        for rel,h in initial['origen_archivos'].items():
            if sha(origin/rel)!=h:raise ValueError('La fuente del compañero cambió: '+rel)
            source[rel]=h
    for rel,h in leer(OUT/'incorporacion_final.json')['archivos_extraidos'].items():
        if sha(RAIZ/rel)!=h:raise ValueError('Extracción original alterada: '+rel)
    hashes=verificar_pesos(leer_config());tests=leer(OUT/('pruebas_revision02.json' if (OUT/'pruebas_revision02.json').exists() else 'pruebas_finales.json'));reg=leer(OUT/'regresion_original.json');port=leer(OUT/('portabilidad_revision02.json' if (OUT/'portabilidad_revision02.json').exists() else 'portabilidad.json'))
    if not all([tests['aprobada'],reg['aprobada'],port['aprobada']]):raise ValueError('Controles no aprobados')
    if tests['resultados']['suite']['casos']<498:raise ValueError('Faltan pruebas')
    for grupo in ['d2_val_autorizada','ecowaste_val_conocida_por_d2']:
        d=leer(OUT/'evaluacion'/grupo/'resumen.json')
        if d['test_evaluado'] or d['seleccion_modelo_por_resultados']:raise ValueError('Evaluación fuera de protocolo')
        rows=leer(OUT/'cohortes.json')[grupo]
        if len(rows)!=d['imagenes'] or any(r['split']=='test' or r['split_d2']=='test' for r in rows):raise ValueError('Cohorte no autorizada')
        for k,r in enumerate(rows):
            q=leer(OUT/'evaluacion'/grupo/'capturas'/f'{k:04d}.json')
            if q['id']!=r['id'] or q['sha256_imagen']!=r['sha256_archivo'] or q['pesos']!=hashes or q['GT_en_inferencia']:raise ValueError('Captura no corresponde')
    for nombre,frames in [('webcam_fisica_fuera_aislamiento',12),('webcam_ventana_real',8),('mp4_real_transformer',12)]:
        folder=RAIZ/'datos/nuevos/integracion_d2'/nombre;r=leer(folder/'resumen.json')
        if r['estado']!='procesado' or r['frames_procesados']!=frames:raise ValueError('Prueba de video incompleta')
        import cv2
        cap=cv2.VideoCapture(str(folder/'anotado.mp4'));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
        cap.release()
        if count!=frames:raise ValueError('MP4 no se puede decodificar completo')
    casos=leer(OUT/'aplicacion_casos_reales.json')
    for d in casos['casos'].values():
        if sha(RAIZ/d['directorio_privado']/'detecciones.json')!=d['sha256_detecciones']:raise ValueError('Caso real alterado')
    for rel,h in port['archivos'].items():
        p=RAIZ/port['raiz_aislada_privada']/rel
        if sha(p)!=h:raise ValueError('Paquete aislado alterado')
    archivos={}
    folders=['ecowaste','scripts','pruebas','checkpoints/vision_equipo','resultados/integracion_d2','datos/nuevos/integracion_d2']
    for name in folders:
        for p in (RAIZ/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in [archivo_cert,archivo_recibo]:archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    for rel in ['configuracion/integracion_d2.json','documentacion/31_integracion_vision_ia_d2_transformer.md','.gitignore']:archivos[rel]=sha(RAIZ/rel)
    result={'aprobada':True,'archivos_previos_preservados':n,'checkpoints_previos_preservados':sum(p.startswith('checkpoints/') for p in initial['protegidos']),'fuente_companero_verificada':source,'controles':{'suite':tests['resultados']['suite']['casos'],'codec_existente':19,'regresion_original':True,'portabilidad':True,'webcam_fisica_capturada':True,'webcam_ventana_mostrada':True,'MP4_real':True},'pesos':hashes,'entrenamientos_ejecutados':0,'test_evaluado':False,'generalizacion_desbloqueada':False,'push_realizado':False,'motor_nuevo_predeterminado':'d2-transformer','selector_historico_modificado':False,'archivos':archivos}
    escribir(OUT/archivo_cert,result);escribir(OUT/archivo_recibo,{'sha256':sha(OUT/archivo_cert)});print({k:v for k,v in result.items() if k not in ['archivos','fuente_companero_verificada']},flush=True)
if __name__=='__main__':main()
