"""Editor prioritario CPU: mismo proyecto humano, guardados recuperables y sin entrenamiento."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sys
import threading
import traceback
from urllib.parse import parse_qs,urlsplit
import uuid
import webbrowser
import urllib.request
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.auxiliares_publicos import Taller,MATERIALES
from ecowaste.asistente_sam2 import AsistenteSAM2
from ecowaste.correcciones_v28_3 import GestionCorrecciones,VERSION,auditorias_verificadas
from ecowaste.incorporacion import leer,publicar,sha,digest
from scripts.anotar_v27_2 import PROYECTO
from scripts.auditar_v28_3 import gestion as gestion_real,SALIDA
ESTATICOS=RAIZ/'interfaz/anotacion_v28_3'


def servidor(taller,motor,puerto=0,gestion=None,salida_auditorias=None):
    g=gestion or GestionCorrecciones(taller,taller.path.parent/'gestion_fixture',[])
    auditorias=Path(salida_auditorias or g.path/'auditorias');csrf=secrets.token_urlsafe(32)
    vistas={};lock=threading.RLock();estado_audit={'ocupada':False,'error':None}
    def tarea(iid,modo='asistida'):
        t=taller.tarea(iid,independiente=modo=='independiente');evs=taller.diario(iid)
        ult=next((e for e in reversed(evs) if e['modo']==modo),None)
        token=secrets.token_urlsafe(24)
        with lock:
            if len(vistas)>2000:vistas.pop(next(iter(vistas)))
            vistas[token]={'id':iid,'modo':t['modo'],'base':t['revision_base'],'revision':t['revision']}
        return {**t,'token_vista':token,'revision_huella':(ult or {}).get('huella'),
            'cobertura_completa':bool(ult and ult['cobertura_completa']),
            'persona_ultima':(ult or {}).get('persona_id'),'fecha_ultimo_guardado':(ult or {}).get('registrado_UTC')}
    def auditar():
        try:
            p,r=g.auditar(auditorias)
            with lock:estado_audit.update(ocupada=False,error=None,archivo=str(p),resumen=r)
        except Exception as e:
            with lock:estado_audit.update(ocupada=False,error=str(e))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):print(fmt%args,flush=True)
        def enviar(self,b,tipo='application/json; charset=utf-8',status=200):
            self.send_response(status)
            for k,v in [('Content-Type',tipo),('Content-Length',str(len(b))),('Cache-Control','no-store'),
                ('X-Content-Type-Options','nosniff'),('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")]:self.send_header(k,v)
            self.end_headers();self.wfile.write(b)
        def json(self,d,status=200):self.enviar(json.dumps(d,ensure_ascii=False,allow_nan=False).encode(),status=status)
        def seguridad(self,post=False):
            hs=[f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
            if self.headers.get('Host') not in hs:raise ValueError('Host local requerido')
            if post:
                if self.headers.get('Origin') not in [None,*['http://'+h for h in hs]]:raise ValueError('Origen externo rechazado')
                if self.headers.get('X-Eco-Token')!=csrf:raise ValueError('Token local caducado o incorrecto; conservar borrador y recargar contexto')
                if self.headers.get_content_type()!='application/json':raise ValueError('JSON requerido')
        def error(self,e,d=None,respaldo=None,status=400):
            eid='err_'+uuid.uuid4().hex
            r={'error':f'{type(e).__name__}: {e}','id_error':eid,'ruta':urlsplit(self.path).path,
                'imagen_id':d.get('imagen_id') if isinstance(d,dict) else None,
                'id_solicitud':d.get('id_solicitud') if isinstance(d,dict) else None,
                'borrador_emergencia_servidor':respaldo,'revision_humana_creada_por_error':False,
                'accion':'Conserva el borrador. Corrige el objeto indicado o recarga solo el contexto si hay conflicto; no sobrescribir otra revisión.'}
            try:publicar(g.path/'errores'/(eid+'.json'),r)
            except OSError:r['error_registro_local']='No se pudo persistir el diagnóstico; descargar copia de emergencia del navegador'
            self.json(r,status)
        def do_GET(self):
            try:
                self.seguridad();u=urlsplit(self.path);q=parse_qs(u.query)
                if u.path in ['/','/app.js','/estilo.css','/recuperacion.js']:
                    n,t={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),
                        '/estilo.css':('estilo.css','text/css; charset=utf-8'),'/recuperacion.js':('recuperacion.js','text/javascript; charset=utf-8')}[u.path]
                    self.enviar((ESTATICOS/n).read_bytes(),t)
                elif u.path=='/api/config':
                    actores={e['persona_id'] for i in taller.imagenes for e in taller.diario(i)}
                    self.json({'version':VERSION,'token':csrf,'materiales':MATERIALES,'dispositivo':'cpu','GPU_utilizada':False,
                        'proyecto_sha256':taller.base['huella'],'entrenamiento':False,'persona_sugerida':next(iter(actores)) if len(actores)==1 else ''})
                elif u.path=='/api/lista':self.json([{'id':i,'fuente':im['fuente'],'estado':taller.tarea(i)['estado']} for i,im in taller.imagenes.items()])
                elif u.path=='/api/tarea':self.json(tarea(q['id'][0],q.get('modo',['asistida'])[0]))
                elif u.path=='/api/imagen':
                    p=taller.archivo_imagen(q['id'][0]);self.enviar(p.read_bytes(),'image/png' if p.suffix.lower()=='.png' else 'image/jpeg')
                elif u.path=='/api/prioridades':self.json({'incidencias':g.listar(),'entrenamiento_autorizado':False})
                elif u.path=='/api/recibo':
                    from ecowaste.incorporacion import identificador
                    p=g.path/'recibos'/(identificador(q['id'][0])+'.json')
                    self.json({'encontrado':p.exists(),'recibo':leer(p) if p.exists() else None})
                elif u.path=='/api/auditoria':
                    ps=auditorias_verificadas(auditorias);r=leer(ps[-1]) if ps else None
                    actual={i:(next((e for e in reversed(taller.diario(i)) if e['modo']=='asistida'),{}) or {}).get('huella') for i in taller.imagenes}
                    vigente=False
                    if ps:
                        imgs=leer(ps[-1].parent/'imagenes.json');incs=leer(ps[-1].parent/'incidencias.json')
                        # El orden de presentación no cambia el estado de las incidencias.
                        vigente=r['codigo_qa']==g.codigo_qa() and all(actual[f['imagen']]==f['revision_huella'] for f in imgs) and digest(sorted(incs,key=lambda x:x['id']))==digest(sorted(g.listar(),key=lambda x:x['id']))
                    self.json({**estado_audit,'resumen':r,'vigente':vigente})
                elif u.path=='/api/borradores':
                    iid=q['id'][0];taller.imagen(iid);carp=g.path/'borradores'/digest(iid)[:24]
                    if 'archivo' in q:
                        nombre=q['archivo'][0]
                        if not re.fullmatch(r'[a-f0-9]{32}\.json',nombre):raise ValueError('Nombre de borrador inválido')
                        self.json(leer(carp/nombre))
                    else:self.json([{'archivo':p.name,'sha256':sha(p)} for p in sorted(carp.glob('*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:10]])
                else:self.json({'error':'Ruta inexistente'},404)
            except (ValueError,KeyError,TypeError,OSError) as e:self.error(e)
        def do_POST(self):
            d=None;respaldo=None
            try:
                self.seguridad(True);n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=12*1024*1024:raise ValueError('Tamaño de solicitud inválido (máximo12MiB)')
                def invalido(x):raise ValueError('Número JSON no finito: '+x)
                d=json.loads(self.rfile.read(n),parse_constant=invalido)
                if not isinstance(d,dict):raise ValueError('Objeto JSON requerido')
                ruta=urlsplit(self.path).path
                if ruta=='/api/guardar':
                    if d.get('proyecto_sha256')!=taller.base['huella']:raise ValueError('Solicitud de otro proyecto')
                    respaldo=g.borrador(d)
                    with lock:
                        r=g.guardar(d,vistas.get(d.get('token_vista')))
                        vistas.pop(d.get('token_vista'),None)
                    self.json({**r,'borrador_emergencia_servidor':respaldo,'tarea':tarea(d['imagen_id'],r['modo'])})
                elif ruta=='/api/borrador':self.json(g.borrador(d))
                elif ruta=='/api/incidencia':self.json(g.resolver(d))
                elif ruta=='/api/auditar':
                    with lock:
                        if not estado_audit['ocupada']:
                            estado_audit.update(ocupada=True,error=None);threading.Thread(target=auditar,daemon=True).start()
                    self.json({'iniciada':True,'entrenamiento':False},202)
                elif ruta=='/api/sam':
                    vista=vistas.get(d.get('token_vista'))
                    if not vista or vista['id']!=d.get('imagen_id'):raise ValueError('Vista inválida; recuperar contexto sin descartar borrador')
                    self.json(motor.proponer(taller,d['imagen_id'],d['indicaciones']))
                elif ruta=='/api/liberar':motor.liberar();self.json({'liberado':True,'GPU_utilizada':False})
                elif ruta=='/api/exportar':self.json(taller.exportar_coco())
                else:self.json({'error':'Ruta inexistente'},404)
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:self.error(e,d,respaldo)
            except Exception as e:self.error(e,d,respaldo,500)
    app=ThreadingHTTPServer(('127.0.0.1',puerto),Handler);app.gestion=g;return app


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--puerto',type=int,default=8768)
    p.add_argument('--proyecto',type=Path,default=PROYECTO);p.add_argument('--abrir',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    t=Taller(a.proyecto,RAIZ)
    if a.abrir:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{a.puerto}/api/config',timeout=2) as r:cfg=json.load(r)
        except OSError:cfg=None
        if cfg:
            if cfg.get('version')!=VERSION or cfg.get('proyecto_sha256')!=t.base['huella']:raise ValueError('Puerto ocupado por otra versión/proyecto')
            webbrowser.open(f'http://127.0.0.1:{a.puerto}');return
    app=servidor(t,AsistenteSAM2(RAIZ),a.puerto,gestion_real(t),SALIDA/'auditorias')
    print(f'Editor V28.3 http://127.0.0.1:{app.server_port} | CPU | sin entrenamiento',flush=True)
    if a.abrir:webbrowser.open(f'http://127.0.0.1:{app.server_port}')
    try:app.serve_forever()
    finally:app.server_close()


if __name__=='__main__':main()
