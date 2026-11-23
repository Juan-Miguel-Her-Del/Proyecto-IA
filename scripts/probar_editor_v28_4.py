"""Pruebas Edge en fixtures; el proyecto real sólo se consulta al final."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import tempfile
import threading
import time
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import numpy as np
from playwright.sync_api import sync_playwright
from ecowaste.incorporacion import publicar,sha,leer
from ecowaste.auxiliares_publicos import rle_coco
from ecowaste.ajustes_ia_v28_3 import decodificar
from ecowaste.revision_ia_v28_4 import sello
from scripts.anotar_v28_4 import servidor
from pruebas.test_revision_ia_v28_4 import crear_fixture


class SAMFixture:
    def proponer(self,t,iid,indicaciones):
        m=np.zeros((48,64),np.uint8);m[5:18,42:60]=1;m[8:12,46:51]=0
        return t.registrar_ia(iid,{'rle_ia_original':rle_coco(m),'segundos':0,'dispositivo':'cpu'},indicaciones,
            {'nombre':'FIXTURE_DESCARTABLE_NO_SAM_REAL','device':'cpu'})
    def liberar(self):pass


def main():
    salida=RAIZ/'resultados/v28_4_revision_IA/pruebas_navegador'/sello();casos=[];errores=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
        def prueba(nombre,fn):
            with tempfile.TemporaryDirectory() as td:
                rev=crear_fixture(td);rev.auditar();app=servidor(rev,motor=SAMFixture());h=threading.Thread(target=app.serve_forever,daemon=True);h.start()
                ctx=browser.new_context(accept_downloads=True,viewport={'width':1500,'height':1000});page=ctx.new_page()
                page.on('pageerror',lambda e:errores.append(str(e)));page.on('dialog',lambda d:d.accept())
                try:
                    page.goto(f'http://127.0.0.1:{app.server_port}/?imagen=fixture%3A1',wait_until='networkidle')
                    page.wait_for_function('()=>Boolean(tarea && uid && !busy)')
                    fn(page,rev,ctx)
                    casos.append(nombre)
                except Exception:
                    carpeta=RAIZ/'datos/nuevos/v28_4/capturas_privadas';carpeta.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(carpeta/(sello()+'_fallo.png')),full_page=True)
                    raise
                finally:
                    ctx.close();app.shutdown();app.detener_auditor();app.server_close();h.join(timeout=5)
        def abrir(p,uid):
            p.locator(f'[data-unidad="{uid}"] button').click();p.wait_for_function('(id)=>uid===id&&!busy',arg=uid)
        def nueva(p):return p.evaluate('tarea.unidades.find(u=>u.objetos.includes("nuevo")).id')
        def wait_ev(p,rev,n):
            p.wait_for_function('()=>!busy && !pendiente && document.getElementById("mensaje").textContent.includes("Guardado confirmado")')
            assert len(rev.eventos())==n
        def lote(p,rev,ctx):
            ids=p.evaluate('tarea.unidades.map(u=>u.id)')
            for uid in ids:abrir(p,uid);p.locator('#inspeccionada').check()
            assert len(rev.eventos())==0
            p.locator('#aceptar').click();wait_ev(p,rev,1)
            assert len(rev.eventos()[0]['decisiones'])==2 and not rev.estado('fixture:1')['cobertura_completa']
            assert not p.locator('#cobertura').is_checked()
            p.locator('#cobertura').check();p.locator('#nota_cobertura').fill('Inspeccioné los objetos y el ignore del fixture completo.')
            p.locator('#confirmar').click();wait_ev(p,rev,2)
            assert rev.estado('fixture:1')['cobertura_completa']
            p.wait_for_function('async()=>Boolean((await api("/api/auditoria")).vigente)')
            with p.expect_download() as dl:p.locator('#exportar').click()
            co=json.loads(Path(dl.value.path()).read_text(encoding='utf8'))
            assert len(co['images'])==2 and any(a['ignore']==1 for a in co['annotations'])
            nuevo=next(a for a in co['annotations'] if a['eco_objeto_id']=='nuevo')
            assert not decodificar(nuevo['segmentation'])[9,48] and not nuevo['eco']['base_humana_anterior']
        prueba('Aceptación múltiple sólo de propuestas inspeccionadas; cobertura separada',lote)
        def editar(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check();p.locator('#material').select_option('carton')
            p.locator('#herramienta').select_option('borrar');p.locator('#radio').evaluate('(e)=>{e.value=1;e.dispatchEvent(new Event("input"))}')
            box=p.locator('#despues').bounding_box();p.mouse.click(box['x']+box['width']*43/64,box['y']+box['height']*6/48)
            p.locator('#nota').fill('Corregí material y un borde con el pincel del fixture.')
            p.locator('#editar').click();wait_ev(p,rev,1)
            a=next(a for a in rev.estado('fixture:1')['anotaciones'] if a['id']=='nuevo')
            assert a['material']=='carton' and not decodificar(a['segmentacion_canonica_RLE'])[9,48]
        prueba('Edición visual de material y máscara RLE conserva el hueco',editar)
        def cuatrocientos(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check()
            p.evaluate('recordar(); arrays.get(activo().id).fill(0); cambiar()')
            p.locator('#nota').fill('Intento con máscara vacía de fixture para verificar recuperación.')
            p.locator('#editar').click();p.wait_for_function('()=>!busy && document.getElementById("mensaje").textContent.includes("Máscara vacía")')
            assert not rev.eventos();assert p.evaluate('pendiente===null && dirty')
            p.locator('#deshacer').click();p.locator('#editar').click();wait_ev(p,rev,1)
        prueba('HTTP400 explica el objeto, conserva borrador y permite corregir y guardar',cuatrocientos)
        def recuperar(p,rev,ctx):
            uid=nueva(p);abrir(p,uid);p.locator('#material').select_option('metal');p.locator('#nota').fill('Borrador de fixture para recuperación tras recargar.')
            p.evaluate('persistir()');p.reload(wait_until='networkidle');p.wait_for_function('()=>Boolean(tarea && uid && !busy)')
            assert not rev.eventos();p.locator('#recuperar').click();p.wait_for_function('()=>Boolean(dirty && activo().material==="metal")')
            assert p.evaluate('uid')==uid and not rev.eventos()
        prueba('Recuperación explícita IndexedDB tras recarga sin crear revisión',recuperar)
        def offline(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#material').select_option('vidrio');ctx.set_offline(True)
            with p.expect_download() as dl:p.locator('#emergencia').click()
            doc=json.loads(Path(dl.value.path()).read_text(encoding='utf8'))
            assert doc['estado']['anotaciones'][0]['material']=='vidrio' and not rev.eventos();ctx.set_offline(False)
        prueba('Copia de emergencia descargable sin red',offline)
        def perdida(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check()
            def perder(route):route.fetch();route.abort('failed')
            p.route('**/api/guardar',perder);p.locator('#aceptar').click()
            p.wait_for_function('()=>Boolean(!busy && pendiente!==null && !document.getElementById("error_detalle").hidden)');assert len(rev.eventos())==1
            p.unroute('**/api/guardar');p.locator('#reintentar').click();wait_ev(p,rev,1)
        prueba('Respuesta perdida después del commit: reintento sin duplicar evento',perdida)
        def doble(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check()
            p.evaluate('const persistirOriginal=persistir;persistir=async()=>{await new Promise(r=>setTimeout(r,250));return persistirOriginal();}')
            p.evaluate('document.getElementById("aceptar").click();document.getElementById("rechazar").click();')
            assert p.evaluate('busy===true')
            wait_ev(p,rev,1);assert rev.eventos()[0]['accion']=='aceptar'
        prueba('Bloqueo inmediato de doble clic antes de esperar IndexedDB',doble)
        def conflicto(p,rev,ctx):
            p2=ctx.new_page();p2.on('dialog',lambda d:d.accept());p2.goto(p.url,wait_until='networkidle');p2.wait_for_function('()=>Boolean(tarea && uid && !busy)')
            for pg in [p,p2]:abrir(pg,nueva(pg));pg.locator('#inspeccionada').check()
            p.locator('#aceptar').click();wait_ev(p,rev,1)
            p2.locator('#nota').fill('Rechazo de fixture desde una pestaña desactualizada.');p2.locator('#rechazar').click()
            p2.wait_for_function('()=>Boolean(!busy && document.getElementById("mensaje").textContent.includes("Conflicto"))')
            assert len(rev.eventos())==1
        prueba('Dos pestañas: la decisión antigua no sobrescribe la nueva',conflicto)
        def sam(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#herramienta').select_option('positivo');b=p.locator('#despues').bounding_box();p.mouse.click(b['x']+b['width']*.7,b['y']+b['height']*.15)
            p.locator('#sam').click();p.wait_for_function('()=>Boolean(!busy && dirty && activo().sam_v284_id)')
            assert not rev.eventos();p.locator('#inspeccionada').check();p.locator('#nota').fill('Corrección humana con SAM simulado exclusivamente en fixture.')
            p.locator('#editar').click();wait_ev(p,rev,1)
            assert len(list((rev.path/'sam').glob('*.json')))==1
        prueba('SAM de fixture sólo modifica borrador hasta guardar edición humana',sam)
        def rechazar(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check();p.locator('#nota').fill('Propuesta nueva rechazada explícitamente en fixture.')
            p.locator('#rechazar').click();wait_ev(p,rev,1)
            assert 'nuevo' not in {a['id'] for a in rev.estado('fixture:1')['anotaciones']}
        prueba('Rechazo conserva la base; no acepta la propuesta ni completa imagen',rechazar)
        def posponer(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check();p.locator('#nota').fill('Necesito revisar más tarde el fixture de este objeto.')
            p.locator('#posponer').click();wait_ev(p,rev,1);assert rev.estado('fixture:1')['pendientes']==2
            assert p.locator('#confirmar').is_disabled()
        prueba('Revisión posterior queda pendiente y bloquea cobertura',posponer)
        def sesion(p,rev,ctx):
            abrir(p,nueva(p));p.locator('#inspeccionada').check();p.locator('#material').select_option('papel')
            p.locator('#nota').fill('Prueba de recuperación tras token local caducado.');p.evaluate('cfg.token="invalido"');p.locator('#editar').click()
            p.wait_for_function('()=>!busy && document.getElementById("mensaje").textContent.includes("Sesión/origen inválido")')
            assert not rev.eventos() and p.evaluate('dirty');p.locator('#contexto').click();p.wait_for_function('()=>document.getElementById("mensaje").textContent.includes("Sesión actualizada")')
            p.locator('#editar').click();wait_ev(p,rev,1)
        prueba('Sesión caducada: contexto renovado conserva el borrador',sesion)
        assert not errores,errores
        # Fotografías reales: GET solamente; no se toca ninguna decisión humana.
        ctx=browser.new_context(viewport={'width':1600,'height':1100});p=ctx.new_page();posts=[]
        p.on('request',lambda r:posts.append(r.url) if r.method!='GET' else None)
        p.goto('http://127.0.0.1:8771/?imagen=hpi%3A23',wait_until='networkidle');p.wait_for_function('()=>Boolean(tarea && uid && !busy)',timeout=60000)
        assert p.locator('#antes').evaluate('(e)=>e.width>0') and p.locator('#despues').evaluate('(e)=>e.width>0')
        assert not p.locator('#inspeccionada').is_checked() and not p.locator('#cobertura').is_checked()
        priv=RAIZ/'datos/nuevos/v28_4/capturas_privadas';priv.mkdir(parents=True,exist_ok=True);p.screenshot(path=str(priv/'editor_real_preparado.png'),full_page=True)
        assert not posts,posts
        ctx.close();casos.append('Proyecto real: RLE antes/después visible; ninguna decisión POST ni casilla preaprobada')
        browser.close()
    publicar(salida/'prueba_editor.json',{'version':'prueba_editor_v284_1','aprobada':True,'casos':casos,
        'errores_js':errores,'escrituras_solo_fixtures':True,'POST_proyecto_real':posts,
        'SAM_real_ejecutado':False,'ediciones_humanas_reales_simuladas':0})
    print({'aprobada':True,'casos':len(casos),'recibo':str(salida/'prueba_editor.json')},flush=True)


if __name__=='__main__':main()
