"""Recomprobación sin alterar ni volver a preparar las correcciones del usuario."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha,publicar,candidatos_duplicados
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import revision_humana_final,analizar_imagen
from ecowaste.datos_v28_2 import AuxiliarExperimental
from ecowaste.experimento_v28_2 import SALIDA,CONFIG,validar


def main():
    cfg=leer(CONFIG);validar(cfg);ds=AuxiliarExperimental(RAIZ,RAIZ/cfg['coleccion'])
    rec=leer(RAIZ/'resultados/v28_validacion_anotaciones/respaldo_verificado.json')
    for raiz in [RAIZ/rec['origen'],RAIZ/rec['destino']]:
        for p,h in rec['archivos'].items():
            if sha(raiz/p)!=h:raise ValueError('Revisión o respaldo alterado: '+str(raiz/p))
    taller=Taller(RAIZ/rec['destino'],RAIZ);refs=[];filas=[]
    for r in ds.registros:
        im=taller.imagen(r['id']);ev=revision_humana_final(taller.diario(r['id']))
        if not ev or not ev['cobertura_completa'] or ev['persona_id']!='juan_miguel':raise ValueError('Falta revisión completa')
        q=analizar_imagen(im,ev['anotaciones'],True)
        if q['errores'] or q['avisos_visuales']:raise ValueError('Incidencias en candidata')
        filas.append({'imagen':r['id'],'huella_revision':ev['huella'],'organicos_utilizables_48':q['organicos_utilizables_48'],
            'exhaustividad_no_certificada':True});refs.append({**im,'id':r['id']})
    # Reutilizar certificado histórico train/val, protegido por hashes V28, sin abrir val ni test.
    previo=RAIZ/'resultados/v28_validacion_anotaciones/duplicados_train_val_y_piloto.json'
    cert=leer(previo);sel={r['id'] for r in ds.registros}
    hist=[d for d in cert['historicos'] if d['imagen'] in sel]
    du=candidatos_duplicados(refs,[])
    if hist or du['nuevos']:raise ValueError('Duplicados pendientes de agrupación')
    ckprev=leer(SALIDA/'checkpoints_previos.json')
    hashes={p:sha(RAIZ/p) for p in ckprev};publicar(SALIDA/'sha256_checkpoints_historicos.json',hashes)
    eventos=sum(len(taller.diario(i)) for i in taller.imagenes)
    if eventos!=134 or len(rec['archivos'])!=731:raise ValueError('Cambió el respaldo')
    publicar(SALIDA/'verificacion_previa_datos.json',{'aprobada':True,'eventos_humanos':eventos,'archivos_respaldo':731,
        'imagenes':filas,'duplicados_nuevos':du,'duplicados_historicos_candidatas':hist,
        'certificado_duplicados_historicos':{'archivo':previo.relative_to(RAIZ).as_posix(),'sha256':sha(previo)},
        'agrupacion':'fuente íntegra solo train; sesiones y objetos desconocidos no acreditados',
        'licencias_sha256':sha(RAIZ/cfg['coleccion']/'LICENCIAS_Y_PROCEDENCIA.json'),
        'licencias_alcance':'académico no comercial, atribución preservada; no redistribuir fotos en Git',
        'checkpoint_historicos':len(hashes),'aceptadas_V26':0})
    print('Reverificación aprobada: 28 imágenes; 134 eventos; 731 archivos; '+str(len(hashes))+' checkpoints protegidos',flush=True)


if __name__=='__main__':main()
