"""Verifica evidencia cacheada original frente a su certificado V30, sin inferencia."""
import sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ));OUT=Path(__file__).resolve().parent
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
def main():
    base=RAIZ/'resultados/v30_hibrido_revision02';cert=leer(base/'integridad.json');rec=leer(base/'recibo_integridad.json')
    if sha(base/'integridad.json')!=rec['sha256']:raise ValueError('Certificado V30 alterado')
    metricas=base/'metricas_piloto.json';refs=leer(metricas)['resultados']['Original'];archivos={metricas.relative_to(RAIZ).as_posix():sha(metricas)}
    for dominio in ['historico_train','nuevo_train','desarrollo','externo_retenido']:
        d=refs[dominio]
        p=RAIZ/d['analisis']
        if sha(p)!=d['sha256_analisis']:raise ValueError('Análisis de referencia alterado')
        archivos[p.relative_to(RAIZ).as_posix()]=sha(p);folder=RAIZ/d['captura'];captura=leer(folder/'captura.json');archivos[(folder/'captura.json').relative_to(RAIZ).as_posix()]=sha(folder/'captura.json')
        for nombre,h in captura['archivos'].items():
            q=folder/nombre
            if sha(q)!=h:raise ValueError('Captura histórica alterada: '+nombre)
            archivos[q.relative_to(RAIZ).as_posix()]=h
    # Se verifica la evidencia enumerada; no se renueva el certificado histórico
    # completo ni se ignora el cambio deliberado y respaldado de .gitignore.
    for p,h in archivos.items():
        if p in cert['archivos'] and cert['archivos'][p]!=h:raise ValueError('No coincide con certificado V30: '+p)
    publicar(OUT/'referencia_verificada.json',{'aprobada':True,'inferencia_repetida':False,'certificado_v30_sha256':sha(base/'integridad.json'),'archivos':archivos,'pesos_originales_sha256':sha(RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt'),'clasificador_sha256':sha(RAIZ/'checkpoints/clasificador_v3_regularizado_mejor.pt'),'alcance':'evidencia Original reutilizada; no recertifica otros modelos V30 ni identidad universal entre backends'})
    print('Referencia cacheada verificada',len(archivos),'archivos',flush=True)
if __name__=='__main__':main()
