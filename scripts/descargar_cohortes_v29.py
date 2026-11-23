"""Extraer solo train/val de ZIP oficiales mediante HTTP Range verificable.

No abre ni extrae test. No afirma verificar SHA del ZIP completo: cada miembro
extraído verifica CRC ZIP y conserva SHA256 propio, versión/ETag de origen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import io,json,struct,sys,time,urllib.request,zipfile,zlib,subprocess,uuid,re
from pathlib import Path,PurePosixPath
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,leer,sha
from scripts.fuentes_v29 import DATOS,SALIDA,PACKWISE_ZIP,url_archivo


class Remoto(io.RawIOBase):
    def __init__(self,url,size=None):
        self.url=url;self.pos=0;self.etag=None;self.requests=[]
        self.obtener(0,0);self.size=int(self.requests[-1]['total'])
        if size is not None and self.size!=size:raise ValueError('Tamaño de ZIP distinto')

    def obtener(self,a,b):
        if b-a+1>1024*1024:
            return b''.join(self.obtener(k,min(b,k+1024*1024-1)) for k in range(a,b+1,1024*1024))
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity','Cache-Control':'no-cache','User-Agent':'EcoWasteFormer/academic-dataset-audit'})
        for intento in range(4):
            try:
                with urllib.request.urlopen(req,timeout=25) as r:
                    if r.status!=206:raise OSError('El servidor no respeta HTTP Range; no descargar implícitamente todo')
                    cr=r.headers['Content-Range'];et=r.headers.get('ETag');total=int(cr.split('/')[-1])
                    if not cr.startswith(f'bytes {a}-{b}/'):raise ValueError('Intervalo recibido distinto')
                    if self.etag and et!=self.etag:raise ValueError('Archivo remoto cambió durante la descarga')
                    self.etag=et
                    # Reutilizar el destino publicado de la redirección evita
                    # pedir un ticket Mendeley nuevo por cada miembro.
                    if len(self.requests)==0:self.url=r.url
                    bloques=[];pendiente=b-a+1
                    while pendiente:
                        bloque=r.read(min(pendiente,1024*1024))
                        if not bloque:break
                        bloques.append(bloque);pendiente-=len(bloque)
                    d=b''.join(bloques)
                    if len(d)!=b-a+1:raise ValueError('Respuesta parcial truncada')
                self.requests.append({'inicio':a,'fin':b,'total':total,'etag':et});return d
            except (OSError,TimeoutError):
                if intento==3:raise
                time.sleep(1+intento)

    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def tell(self):return self.pos
    def seekable(self):return True
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        d=self.obtener(self.pos,self.pos+n-1);self.pos+=n;return d


def seguro(nombre):
    p=PurePosixPath(nombre)
    if p.is_absolute() or '..' in p.parts or '\\' in nombre or ':' in nombre:raise ValueError('Ruta ZIP insegura')
    return p


def es_test(nombre):
    parts=[s.lower() for s in PurePosixPath(nombre).parts]
    return any(s in ['test','testing'] or s.startswith(('test.','test_','_test')) or s.endswith(('_test.json','_test.jpg','_test.png')) for s in parts)


def fuente(nombre):
    if nombre=='packwise':
        ficha=leer(sorted(SALIDA.glob('diagnostico_red_*.json'))[-1])
        meta=json.loads(ficha['packwise_rest']['respuesta'])
        fs=[b for b in meta['bitstreams'] if b['name']=='PackWISE_dataset_v2.zip' and b['bundleName']=='ORIGINAL']
        if len(fs)!=1:raise ValueError('Referencia REST PackWISE ambigua')
        b=fs[0]
        return 'https://fordatis.fraunhofer.de'+b['retrieveLink'],b['sizeBytes'],None
    f=leer(next((DATOS/'fichas').glob('*/click_archivos.json')))[0]
    return url_archivo(f),f['size'],f['content_details']['sha256_hash']


def miembro(remoto,info,destino,transporte='urllib'):
    # Miembro completo en una solicitud, incluido su encabezado local.
    seguro(info.filename)
    if es_test(info.filename):raise ValueError('Partición test prohibida')
    if info.flag_bits&1:raise ValueError('ZIP cifrado no admitido')
    if info.file_size>64*1024**2:raise ValueError('Miembro demasiado grande')
    if destino.exists():
        previo=destino.read_bytes()
        if len(previo)!=info.file_size or zlib.crc32(previo)&0xffffffff!=info.CRC:raise ValueError('Miembro previamente extraído alterado')
        return {'ruta_zip':info.filename,'archivo':destino.relative_to(RAIZ).as_posix(),'bytes':len(previo),'sha256':sha(destino),'crc32_zip':f'{info.CRC:08x}','reutilizado':True}
    final=min(remoto.size-1,info.header_offset+30+4096+info.compress_size-1)
    if transporte=='curl':
        cache=DATOS/'click/rangos';cache.mkdir(parents=True,exist_ok=True);key=uuid.uuid4().hex
        archivo=cache/(key+'.partial');cab=cache/(key+'.headers')
        cmd=['curl.exe','--ssl-revoke-best-effort','--silent','--show-error','--location','--fail','--http1.1',
            '--connect-timeout','20','--max-time','120','--retry','2','--retry-all-errors',
            '--max-filesize',str(final-info.header_offset+1),'--range',f'{info.header_offset}-{final}',
            '--dump-header',str(cab),'--output',str(archivo),remoto.url]
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if r.returncode:raise OSError('curl '+str(r.returncode)+': '+r.stderr.decode('utf8',errors='replace')[-400:])
        cs=re.findall(r'Content-Range:\s*bytes (\d+)-(\d+)/(\d+)',cab.read_text(errors='replace'),re.I)
        if not cs or tuple(map(int,cs[-1]))!=(info.header_offset,final,remoto.size):raise ValueError('Intervalo curl incompatible')
        d=archivo.read_bytes()
        if len(d)!=final-info.header_offset+1:raise ValueError('Intervalo curl truncado')
        # Son temporales de transporte propios; los fallidos se conservan.
        archivo.unlink();cab.unlink()
    else:d=remoto.obtener(info.header_offset,final)
    header=struct.unpack('<IHHHHHIIIHH',d[:30])
    if header[0]!=0x04034b50:raise ValueError('Encabezado local inválido')
    inicio=30+header[-2]+header[-1]
    if inicio>4126:raise ValueError('Extra ZIP inusualmente grande')
    payload=d[inicio:inicio+info.compress_size]
    raw=payload if info.compress_type==zipfile.ZIP_STORED else zlib.decompress(payload,-15) if info.compress_type==zipfile.ZIP_DEFLATED else None
    if raw is None:raise ValueError('Compresión ZIP no admitida')
    if len(raw)!=info.file_size or zlib.crc32(raw)&0xffffffff!=info.CRC:raise ValueError('CRC/tamaño del miembro incorrecto')
    destino.parent.mkdir(parents=True,exist_ok=True)
    if destino.exists():
        if destino.read_bytes()!=raw:raise ValueError('Descarga previa distinta; no sobrescribir')
    else:
        with destino.open('xb') as f:f.write(raw)
    return {'ruta_zip':info.filename,'archivo':destino.relative_to(RAIZ).as_posix(),'bytes':len(raw),'sha256':sha(destino),'crc32_zip':f'{info.CRC:08x}'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fuente',choices=['packwise','click'],required=True)
    ap.add_argument('--extraer-lista',type=Path);ap.add_argument('--hilos',type=int,default=6);ap.add_argument('--transporte',choices=['urllib','curl'],default='urllib');a=ap.parse_args()
    print('Leyendo índice oficial',a.fuente,flush=True)
    url,size,decl=fuente(a.fuente);r=Remoto(url,size)
    z=zipfile.ZipFile(r);infos=z.infolist()
    indice={'fuente':a.fuente,'url':url,'bytes_zip':r.size,'etag':r.etag,'sha256_zip_declarado_no_verificado':decl,
        'sha256_zip_verificado':False,'miembros':[{'nombre':i.filename,'bytes':i.file_size,'comprimidos':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in infos]}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=SALIDA/('descarga_'+a.fuente+'_'+stamp)
    publicar(out/'indice.json',indice)
    print('Índice verificado',a.fuente,len(infos),'miembros',flush=True)
    if not a.extraer_lista:
        print('Indice',out/'indice.json',flush=True)
        print(json.dumps(indice['miembros'][:30],indent=2),flush=True);return
    seleccion=leer(a.extraer_lista)
    if not isinstance(seleccion,list) or len(seleccion)!=len(set(seleccion)):raise ValueError('Lista de miembros inválida')
    mapa={i.filename:i for i in infos};recibos=[];fallos=[]
    for n in seleccion:
        seguro(n)
        if es_test(n) or n not in mapa:raise ValueError('Selección prohibida o inexistente: '+n)
    publicar(out/'seleccion.json',seleccion)
    inicio=time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(max(a.hilos,1),8)) as ex:
        futures={ex.submit(miembro,r,mapa[n],DATOS/a.fuente/'extraidos'/n,a.transporte):n for n in seleccion}
        for f in as_completed(futures):
            n=futures[f]
            try:recibos.append(f.result())
            except Exception as e:fallos.append({'miembro':n,'error':type(e).__name__+': '+str(e)})
            if (len(recibos)+len(fallos))%20==0:print(a.fuente,len(recibos),'OK',len(fallos),'fallos /',len(seleccion),flush=True)
    publicar(out/'recibo.json',{'fuente':a.fuente,'licencia':'CC BY 4.0; véanse fichas oficiales',
        'archivos':sorted(recibos,key=lambda d:d['ruta_zip']),'errores':fallos,'segundos':time.perf_counter()-inicio,
        'particion_test_extraida':False,'transporte':a.transporte,'version_remota_etag':r.etag,'rangos':r.requests,'indice_sha256':sha(out/'indice.json')})
    print('Recibo',out/'recibo.json',flush=True)
    if fallos:raise RuntimeError('Miembros fallidos; conservar diagnóstico y no entrenar aún')


if __name__=='__main__':main()
