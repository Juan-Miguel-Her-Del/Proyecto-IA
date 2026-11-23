"""Editor RLE de aprobación humana de IA: proyecto separado, SAM a demanda en CPU."""
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import parse_qs,urlsplit
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.revision_ia_v28_4 import RevisionIA,VERSION
from ecowaste.incorporacion import leer,publicar,sha,digest,identificador
from ecowaste.auxiliares_publicos import Taller,MATERIALES
from ecowaste.asistente_sam2 import AsistenteSAM2
from scripts.anotar_v27_2 import PROYECTO
UI=RAIZ/'interfaz/anotacion_v28_4'


def servidor(rev,puerto=0,motor=None):
    motor=motor or AsistenteSAM2(rev.raiz);csrf=secrets.token_urlsafe(32)
    def huella_backend():return digest({p.name:sha(p) for p in [Path(__file__),Path(sys.modules[RevisionIA.__module__].__file__)]})
    codigo_memoria=huella_backend()
    vistas={};inspecciones={};mutex=threading.RLock();solicitud_audit=threading.Event();stop=threading.Event()
    estado_audit={'ocupada':False,'error':None}
    def worker():
        while not stop.is_set():
            if not solicitud_audit.wait(.3):continue
            solicitud_audit.clear()
            with mutex:estado_audit.update(ocupada=True,error=None)
            try:rev.auditar()
            except Exception as e:
                with mutex:estado_audit['error']=str(e)
            finally:
                with mutex:estado_audit['ocupada']=False
    hilo=threading.Thread(target=worker,daemon=True);hilo.start()
    if not rev.ultima_auditoria()['vigente']:solicitud_audit.set()
    def tarea(iid):
        st=rev.tarea(iid);token=secrets.token_urlsafe(24)
        with mutex:
            if len(vistas)>2000:vistas.clear();inspecciones.clear()
            vistas[token]={'imagen':iid,'estado':st['estado_sha256']}
        return {**st,'token_vista':token}
    class DestinoSAM:
        def imagen(self,iid):return rev.imagenes[iid]['imagen']
        def archivo_imagen(self,iid):
            im=self.imagen(iid);path=rev.raiz/im['archivo']
            if sha(path)!=im['sha256_archivo']:raise ValueError('Imagen alterada')
            return path
        def registrar_ia(self,iid,resultado,indicaciones,modelo):
            r={'id':'sam_v284_'+uuid.uuid4().hex,'proyecto_sha256':rev.id,'imagen_id':iid,
               'resultado':resultado,'indicaciones':indicaciones,'modelo':modelo,
               'registrado_UTC':datetime.now(timezone.utc).isoformat(),'revision_humana':False,'aceptada_V26':False}
            r['huella']=digest(r);publicar(rev.path/'sam'/(r['id']+'.json'),r);return r
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):print(fmt%args,flush=True)
        def enviar(self,blob,tipo='application/json; charset=utf-8',status=200,extra=None):
            self.send_response(status)
            for k,v in [('Content-Type',tipo),('Content-Length',str(len(blob))),('Cache-Control','no-store'),
                ('X-Content-Type-Options','nosniff'),('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'"),*(extra or [])]:self.send_header(k,v)
            try:self.end_headers();self.wfile.write(blob)
            except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):self.close_connection=True
        def json(self,d,status=200):self.enviar(json.dumps(d,ensure_ascii=False,allow_nan=False).encode(),status=status)
        def seguridad(self,post=False):
            hosts=[f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
            if self.headers.get('Host') not in hosts:raise ValueError('Host local requerido')
            if post:
                if self.headers.get('Origin') not in [None,*['http://'+h for h in hosts]] or self.headers.get('X-Eco-Token')!=csrf:raise ValueError('Sesión/origen inválido; conservar borrador y recargar contexto')
                if self.headers.get_content_type()!='application/json':raise ValueError('JSON requerido')
        def error(self,e,status=400,copia=None,commit=None):
            self.json({'error':type(e).__name__+': '+str(e),'id_error':'err_'+uuid.uuid4().hex,
                'borrador_servidor':copia,'decision_guardada':commit is not None,
                'evento_confirmado':commit['evento']['huella'] if commit else None,
                'accion':'Consulta el recibo de la solicitud; el evento sí quedó guardado.' if commit else 'Conserva/descarga tu borrador. No se creó una decisión con esta solicitud fallida.'},status)
        def do_GET(self):
            try:
                self.seguridad();u=urlsplit(self.path);q=parse_qs(u.query)
                if u.path in ['/','/app.js','/estilo.css']:
                    n,t={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/estilo.css':('estilo.css','text/css; charset=utf-8')}[u.path]
                    self.enviar((UI/n).read_bytes(),t)
                elif u.path=='/api/config':self.json({'version':VERSION,'token':csrf,'proyecto_sha256':rev.id,'materiales':MATERIALES,
                    'persona_sugerida':rev.catalogo.get('personas_historicas',[''])[0],'entrenamiento':False,'GPU_utilizada':False,
                    'version_IA_origen':rev.catalogo['version_IA_origen'],'backend_en_memoria_sha256':codigo_memoria,
                    'reinicio_necesario':codigo_memoria!=huella_backend()})
                elif u.path=='/api/lista':
                    evs=rev.eventos();self.json([{'id':iid,'propuestas':len(r['unidades']),
                        'pendientes':(st:=rev.estado(iid,evs))['pendientes'],'completa':st['cobertura_completa'],
                        'base_heredada':r['candidata_heredada_V283'],'origen_vigente':st['origen_vigente']} for iid,r in rev.imagenes.items()])
                elif u.path=='/api/tarea':self.json(tarea(q['id'][0]))
                elif u.path=='/api/inspeccion':
                    iid=q['id'][0];uid=q['unidad'][0];st=rev.estado(iid)
                    un=next(x for x in st['unidades'] if x['id']==uid);token=secrets.token_urlsafe(24)
                    with mutex:inspecciones[token]={'imagen':iid,'unidad':uid,'estado':st['estado_sha256'],'huella':un['huella']}
                    self.json({'token_inspeccion':token,'unidad_id':uid,'huella':un['huella'],'estado_sha256':st['estado_sha256']})
                elif u.path=='/api/imagen':
                    im=rev.imagenes[q['id'][0]]['imagen'];p=rev.raiz/im['archivo']
                    if sha(p)!=im['sha256_archivo']:raise ValueError('Fotografía alterada')
                    self.enviar(p.read_bytes(),'image/png' if p.suffix.lower()=='.png' else 'image/jpeg')
                elif u.path=='/api/auditoria':self.json({**rev.ultima_auditoria(),**estado_audit})
                elif u.path=='/api/diario':self.json(rev.eventos())
                elif u.path=='/api/recibo':
                    sol=q['id'][0];en=next((e for e in rev.eventos() if e['id_solicitud']==sol),None)
                    self.json({'encontrado':en is not None,'evento':en})
                elif u.path=='/api/descarga':
                    carpeta=q['version'][0];identificador(carpeta);p=rev.path/'exportaciones'/carpeta/'annotations.json'
                    self.enviar(p.read_bytes(),extra=[('Content-Disposition','attachment; filename="ecowaste_v28_4_COCO_NO_ENTRENAR.json"')])
                else:self.json({'error':'Ruta inexistente'},404)
            except (ValueError,KeyError,TypeError,OSError,StopIteration) as e:self.error(e)
        def do_POST(self):
            copia=None;commit=None
            try:
                self.seguridad(True);n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=32*1024*1024:raise ValueError('Tamaño de solicitud inválido (máximo32MiB)')
                def invalido(x):raise ValueError('Número JSON no finito: '+x)
                d=json.loads(self.rfile.read(n),parse_constant=invalido)
                if not isinstance(d,dict):raise ValueError('Objeto JSON requerido')
                ruta=urlsplit(self.path).path
                if ruta=='/api/guardar':
                    if d.get('proyecto_sha256')!=rev.id:raise ValueError('Otro proyecto')
                    publicar(rev.path/'borradores_servidor'/(c:=uuid.uuid4().hex+'.json'),{'contenido':d,'no_es_revision':True});copia=c
                    existente=any(e['id_solicitud']==d.get('id_solicitud') for e in rev.eventos())
                    if not existente:
                        if codigo_memoria!=huella_backend():raise ValueError('El código del editor cambió: reiniciar el servicio, conservando el borrador')
                        v=vistas.get(d.get('token_vista'))
                        if not v or v!={'imagen':d.get('imagen_id'),'estado':d.get('estado_sha256')}:raise ValueError('Contexto caducado; recuperar sin perder borrador')
                        if d.get('accion')!='confirmar_cobertura':
                            for uid in d.get('unidades',[]):
                                ins=inspecciones.get(d.get('tokens_inspeccion',{}).get(uid))
                                if not ins or ins['imagen']!=d['imagen_id'] or ins['unidad']!=uid or ins['estado']!=d['estado_sha256']:raise ValueError('Abrir e inspeccionar cada propuesta antes de decidir')
                    r=rev.guardar(d);commit=r;solicitud_audit.set();self.json({**r,'tarea':tarea(d['imagen_id']),'auditoria_pendiente':True})
                elif ruta=='/api/auditar':solicitud_audit.set();self.json({'iniciada':True,'no_es_decision_humana':True},202)
                elif ruta=='/api/exportar':
                    carpeta,coco=rev.exportar(d.get('solo_utilizables',True) is True)
                    self.json({'descarga':'/api/descarga?version='+carpeta.name,'imagenes':len(coco['images']),
                        'mascaras':len(coco['annotations']),'entrenamiento':False,'aceptadas_V26':0})
                elif ruta=='/api/sam':
                    if codigo_memoria!=huella_backend():raise ValueError('Reiniciar servicio antes de ejecutar SAM; conservar borrador')
                    rev.verificar_fuentes()
                    v=vistas.get(d.get('token_vista'))
                    if not v or v['imagen']!=d.get('imagen_id') or v['estado']!=rev.estado(d['imagen_id'])['estado_sha256']:raise ValueError('SAM: contexto antiguo; conserva el borrador')
                    self.json(motor.proponer(DestinoSAM(),d['imagen_id'],d['indicaciones']))
                elif ruta=='/api/liberar':motor.liberar();self.json({'liberado':True})
                else:self.json({'error':'Ruta inexistente'},404)
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:self.error(e,copia=copia,commit=commit)
            except Exception as e:self.error(e,500,copia,commit)
    app=ThreadingHTTPServer(('127.0.0.1',puerto),Handler);app.revision=rev
    app.detener_auditor=lambda:(stop.set(),solicitud_audit.set(),hilo.join(timeout=5))
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proyecto',type=Path);p.add_argument('--puerto',type=int);a=p.parse_args()
    cfg=leer(RAIZ/'configuracion/revision_ia_v28_4.json');rev=RevisionIA(a.proyecto or RAIZ/cfg['proyecto'],RAIZ,Taller(PROYECTO,RAIZ))
    app=servidor(rev,a.puerto or cfg['puerto'])
    print(f'Editor V28.4 http://127.0.0.1:{app.server_port}/ · RLE · SAM CPU a demanda · sin entrenamiento',flush=True)
    try:app.serve_forever()
    finally:app.detener_auditor();app.server_close()


if __name__=='__main__':main()
