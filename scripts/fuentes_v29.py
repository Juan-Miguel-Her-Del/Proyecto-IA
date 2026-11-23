"""Descarga verificable de fuentes públicas PackWISE v2/CLICK v2, sin entrenar.

Sólo URLs públicas prefijadas y archivos ZIP publicados por sus proveedores.
No se ejecuta código descargado ni se extrae ninguna partición test.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import urllib.parse
import uuid
import time
import subprocess
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,sha,leer
DATOS=RAIZ/'datos/nuevos/v29/externos'
SALIDA=RAIZ/'resultados/v29_packwise_click/fuentes'
URLS={
    'packwise_pagina':'https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en',
    'packwise_registro':'https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en&mode=full',
    'click_pagina':'https://data.mendeley.com/datasets/htcvvz6fm2/2',
    'click_meta':'https://data.mendeley.com/public-api/datasets/htcvvz6fm2/versions/2',
    'click_snapshot':'https://api.data.mendeley.com/datasets/htcvvz6fm2/snapshot?version=2',
    'click_archivos':'https://data.mendeley.com/public-api/datasets/htcvvz6fm2/files?folder_id=root&version=2&$start=0&$limit=1000'
}
PACKWISE_ZIP='https://fordatis.fraunhofer.de/bitstream/fordatis/463.2/3/PackWISE_dataset_v2.zip'


def abrir(url):
    if urllib.parse.urlsplit(url).scheme!='https':raise ValueError('HTTPS público requerido')
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EcoWasteFormer/academic-dataset-audit','Accept-Encoding':'identity'}),timeout=90)


def evidencia(carpeta):
    registros={};contenido={}
    for nombre,url in URLS.items():
        try:
            with abrir(url) as r:
                data=r.read(12*1024*1024+1)
                if len(data)>12*1024*1024:raise ValueError('Ficha demasiado grande')
                meta={'url_solicitada':url,'url_final':r.url,'status':r.status,'cabeceras':dict(r.headers),'bytes':len(data)}
            path=DATOS/'fichas'/carpeta.name/(nombre+('.json' if nombre in ['click_meta','click_snapshot','click_archivos'] else '.html'))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            registros[nombre]={**meta,'archivo':path.relative_to(RAIZ).as_posix(),'sha256':sha(path)}
            contenido[nombre]=data.decode('utf8',errors='replace')
            print(nombre,meta['status'],len(data),flush=True)
        except Exception as e:
            registros[nombre]={'url_solicitada':url,'error':type(e).__name__+': '+str(e)}
            print(nombre,type(e).__name__,str(e),flush=True)
    publicar(carpeta/'fichas.json',registros)
    return registros,contenido


def cc_by(texto):
    low=texto.lower().replace('_','-')
    return any(x in low for x in ['creativecommons.org/licenses/by/4.0','cc by 4.0','cc-by-4.0'])


def descargar(url,destino,carpeta,maximo=4*1024**3):
    recibo=destino.with_suffix('.descarga.json')
    if destino.exists():
        old=leer(recibo)
        if sha(destino)!=old['sha256']:raise ValueError('Descarga previa alterada')
        print('Conservada descarga existente',destino.name,flush=True);return old
    destino.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destino.parent).free<maximo+1024**3:raise OSError('Reservar 5GiB libres antes de descargar')
    part=destino.with_suffix('.'+uuid.uuid4().hex[:8]+'.partial');h=hashlib.sha256();md5=hashlib.md5();total=0
    try:
        with abrir(url) as r,part.open('xb') as out:
            esperado=int(r.headers.get('Content-Length','0'));final=r.url;status=r.status
            if esperado>maximo:raise ValueError('Archivo supera presupuesto de descarga')
            while True:
                bloque=r.read(1024*1024)
                if not bloque:break
                total+=len(bloque)
                if total>maximo:raise ValueError('Límite de descarga excedido')
                out.write(bloque);h.update(bloque);md5.update(bloque)
                if total%(32*1024*1024)==0:print(destino.name,total//(1024*1024),'MiB',flush=True)
            if esperado and esperado!=total:raise ValueError('Descarga incompleta')
        import zipfile
        if not zipfile.is_zipfile(part):raise ValueError('El servidor no devolvió un ZIP')
        part.rename(destino)
        meta={'url':url,'url_final':final,'status':status,'archivo':destino.relative_to(RAIZ).as_posix(),
              'bytes':total,'sha256':h.hexdigest(),'md5_calculado':md5.hexdigest(),
              'licencia_fuente_verificada':'CC BY 4.0; conservar fichas/atribución','test_extraido':False}
        publicar(recibo,meta);publicar(carpeta/(destino.stem+'_descarga.json'),meta);return meta
    except Exception as e:
        publicar(carpeta/(destino.stem+'_fallo.json'),{'error':str(e),'archivo_parcial':str(part),'bytes_observados':total})
        raise


def candidatos(x):
    if isinstance(x,dict):
        nombre=x.get('filename') or x.get('name') or x.get('file_name')
        if isinstance(nombre,str) and nombre.lower().endswith('.zip'):
            yield x
        for v in x.values():yield from candidatos(v)
    elif isinstance(x,list):
        for v in x:yield from candidatos(v)


def url_archivo(f):
    for k in ['download_url','downloadUrl','download_link']:
        if f.get(k):return f[k]
    for k in ['content_details','contentDetails','links']:
        if isinstance(f.get(k),dict):
            try:return url_archivo(f[k])
            except ValueError:pass
    raise ValueError('Falta URL de archivo publicado; no inventar descarga')


def sondear_click():
    """Comprobar acceso parcial sin descargar imágenes ni particiones."""
    f=json.loads(next((DATOS/'fichas').glob('*/click_archivos.json')).read_text(encoding='utf8'))[0]
    req=urllib.request.Request(url_archivo(f),headers={'Range':'bytes=0-1023','Accept-Encoding':'identity','User-Agent':'EcoWasteFormer/academic-dataset-audit'})
    with urllib.request.urlopen(req,timeout=90) as r:
        d=r.read(1024)
        meta={'status':r.status,'cabeceras':dict(r.headers),'bytes_leidos':len(d),'firma':d[:4].hex(),'sha256_declarado_archivo':f['content_details']['sha256_hash']}
    out=SALIDA/('sondeo_click_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    publicar(out,meta);print(json.dumps(meta,indent=2),flush=True)


def diagnosticar_red():
    f=json.loads(next((DATOS/'fichas').glob('*/click_archivos.json')).read_text(encoding='utf8'))[0]
    urls={'click':url_archivo(f),'packwise':PACKWISE_ZIP,
        'packwise_rest':'https://fordatis.fraunhofer.de/rest/handle/fordatis/463.2?expand=bitstreams'}
    registros={}
    for nombre,url in urls.items():
        t=time.perf_counter();print('Conectando',nombre,flush=True)
        try:
            req=urllib.request.Request(url,headers={'Range':'bytes=0-65535','Accept':'application/json','User-Agent':'EcoWasteFormer/academic-dataset-audit','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=25) as r:
                print(nombre,'cabeceras',r.status,dict(r.headers),flush=True)
                d=r.read(65536)
                registros[nombre]={'status':r.status,'host_final':urllib.parse.urlsplit(r.url).netloc,'cabeceras':dict(r.headers),'bytes_leidos':len(d),'segundos':time.perf_counter()-t}
                if nombre=='packwise_rest':registros[nombre]['respuesta']=d.decode('utf8',errors='replace')
            print(nombre,registros[nombre]['bytes_leidos'],registros[nombre]['segundos'],flush=True)
        except Exception as e:registros[nombre]={'error':str(e),'segundos':time.perf_counter()-t};print(nombre,registros[nombre],flush=True)
    publicar(SALIDA/('diagnostico_red_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),registros)


def curl_packwise():
    meta=json.loads(leer(sorted(SALIDA.glob('diagnostico_red_*.json'))[-1])['packwise_rest']['respuesta'])
    b=next(x for x in meta['bitstreams'] if x['name']=='PackWISE_dataset_v2.zip' and x['bundleName']=='ORIGINAL')
    target=DATOS/'packwise/PackWISE_dataset_v2.zip'
    if target.exists():print('ZIP ya conservado',sha(target),flush=True);return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');part=target.with_suffix('.curl_'+stamp+'.partial')
    url='https://fordatis.fraunhofer.de'+b['retrieveLink']
    # Schannel no alcanza OCSP en esta máquina. Best-effort conserva validación
    # de cadena/nombre y tolera únicamente la indisponibilidad del servicio OCSP.
    cmd=['curl.exe','--ssl-revoke-best-effort','--location','--fail','--http1.1','--connect-timeout','20','--max-time','600','--retry','3','--retry-all-errors','--output',str(part),url]
    print('Descarga REST oficial mediante curl, MD5 de autores prefijado',flush=True)
    t=time.perf_counter();r=subprocess.run(cmd)
    md5=hashlib.md5(part.read_bytes()).hexdigest() if part.exists() else None
    ok=r.returncode==0 and part.stat().st_size==b['sizeBytes'] and md5==b['checkSum']['value'].strip('"')
    recibo={'archivo_parcial':part.relative_to(RAIZ).as_posix(),'url':url,'bytes':part.stat().st_size if part.exists() else 0,
        'codigo_retorno':r.returncode,'md5_publicado':b['checkSum']['value'].strip('"'),'md5_observado':md5,
        'sha256':sha(part) if part.exists() else None,'aprobada':ok,'segundos':time.perf_counter()-t,'test_extraido':False}
    publicar(SALIDA/('curl_packwise_'+stamp+'.json'),recibo)
    if not ok:raise RuntimeError('ZIP incompleto/incorrecto; conservar evidencia sin entrenar')
    part.rename(target);publicar(target.with_suffix('.descarga.json'),{**recibo,'archivo':target.relative_to(RAIZ).as_posix()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--descargar',action='store_true');p.add_argument('--sondear-click',action='store_true');p.add_argument('--solo-click',action='store_true');p.add_argument('--solo-packwise',action='store_true');p.add_argument('--diagnosticar-red',action='store_true');p.add_argument('--curl-packwise',action='store_true');a=p.parse_args()
    if a.curl_packwise:return curl_packwise()
    if a.diagnosticar_red:return diagnosticar_red()
    if a.sondear_click:return sondear_click()
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];carpeta=SALIDA/marca
    registros,cont=evidencia(carpeta)
    if not a.descargar:return
    if not cc_by(cont.get('packwise_registro','')+cont.get('packwise_pagina','')):raise ValueError('Licencia PackWISE no confirmada')
    if not a.solo_click:descargar(PACKWISE_ZIP,DATOS/'packwise/PackWISE_dataset_v2.zip',carpeta)
    if a.solo_packwise:return
    if not cc_by(cont.get('click_meta','')+cont.get('click_snapshot','')+cont.get('click_pagina','')):raise ValueError('Licencia CLICK no confirmada')
    fs=[]
    for k in ['click_snapshot','click_archivos']:
        try:fs+=list(candidatos(json.loads(cont.get(k,'{}'))))
        except json.JSONDecodeError:pass
    unicos={}
    for f in fs:
        try:unicos[url_archivo(f)]=f
        except ValueError:pass
    publicar(carpeta/'candidatos_archivo_CLICK.json',list(unicos.values()))
    if len(unicos)!=1:raise ValueError('Revisar listado CLICK: '+str(len(unicos))+' ZIP disponibles')
    url=next(iter(unicos));host=urllib.parse.urlsplit(url).netloc
    if not host.endswith('mendeley.com'):raise ValueError('Revisar dominio de descarga CLICK publicado: '+host)
    meta=descargar(url,DATOS/'click/CLICK_dataset_v2.zip',carpeta,maximo=8*1024**3)
    esperado=next(iter(unicos.values()))['content_details'].get('sha256_hash')
    if esperado and meta['sha256']!=esperado:raise ValueError('SHA256 CLICK distinto al publicado por Mendeley')


if __name__=='__main__':main()
