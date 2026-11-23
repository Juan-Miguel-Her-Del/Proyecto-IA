"""Prueba navegador: escritura solo en fixtures; 72 fotos reales estrictamente lectura."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import tempfile
import threading
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from playwright.sync_api import sync_playwright,expect
from pruebas.fixtures_anotacion_v27_2 import fixture
from ecowaste.incorporacion import digest,leer,sha,publicar
from ecowaste.auxiliares_publicos import Taller
from ecowaste.correcciones_v28_3 import GestionCorrecciones
from scripts.anotar_v28_3 import servidor,PROYECTO
from scripts.auditar_v28_3 import gestion,SALIDA
from pruebas.test_correcciones_v28_3 import incidencia,MotorFixture


def iniciar(t,g,m):
    a=servidor(t,m,0,g);h=threading.Thread(target=a.serve_forever,daemon=True);h.start();return a,h,f'http://127.0.0.1:{a.server_port}'


def main():
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];casos=[];errores=[]
    with tempfile.TemporaryDirectory(prefix='editor_v283_fixture_') as td,sync_playwright() as pw:
        r=Path(td);t=fixture(r);base=deepcopy(t.base);im=deepcopy(base['imagenes'][0]);im['id']='fixture:2';base['imagenes'].append(im)
        base['huella']=digest({k:v for k,v in base.items() if k!='huella'});(t.path/'PROYECTO.json').write_text(json.dumps(base),encoding='utf-8');t=Taller(t.path,r)
        inc=incidencia();inc2={**inc,'id':'inc_fixture2','imagen':'fixture:2','motivo':'objeto de la segunda imagen'}
        g=GestionCorrecciones(t,r/'gestion',[inc,inc2]);a,h,url=iniciar(t,g,MotorFixture());b=pw.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
        contexto=b.new_context(accept_downloads=True,viewport={'width':1540,'height':1050});page=contexto.new_page();page.on('pageerror',lambda e:errores.append(str(e)))
        try:
            page.goto(url);expect(page.locator('#objetos option')).to_have_count(1);expect(page.locator('#incidencias option')).to_have_count(2)
            page.fill('#persona','actor_fixture_browser');page.select_option('#material','plastico')
            page.wait_for_function('() => dirty === true');page.evaluate('() => persistir(true)')
            page.select_option('#incidencias','inc_fixture2');expect(page.locator('#titulo')).to_contain_text('fixture:2')
            expect(page.locator('#observacion')).to_contain_text('segunda imagen');page.select_option('#incidencias','inc_fixture')
            expect(page.locator('#recuperar')).to_be_enabled();page.click('#recuperar');expect(page.locator('#material')).to_have_value('plastico')
            casos.append('navegacion_prioritaria_conserva_borrador_por_imagen')
            # Guardado rechazado: geometría incompleta, causa exacta y copia fuera de GT.
            page.evaluate('() => { anns[0].poligonos=[]; dirty=true; }');page.check('#completa');page.click('#guardar')
            expect(page.locator('#error_detalle')).to_contain_text('Objeto 1');expect(page.locator('#error_detalle')).to_contain_text('objeto_1')
            assert not t.diario('fixture:1');assert list((g.path/'borradores').rglob('*.json'))
            casos.append('HTTP400_preciso_y_respaldo_sin_revision_humana')
            with page.expect_download() as dl:page.click('#emergencia')
            ruta=r/'emergencia.json';dl.value.save_as(ruta);em=leer(ruta);assert em['estado']['anotaciones'][0]['poligonos']==[]
            assert not em['estado']['entrenamiento_autorizado'];casos.append('descarga_emergencia_con_geometria_incompleta')
            page.reload();expect(page.locator('#recuperar')).to_be_enabled();page.click('#recuperar')
            assert page.evaluate('() => anns[0].poligonos.length')==0;assert not t.diario('fixture:1')
            casos.append('recuperar_tras_recargar_no_aprueba_anotacion')
            # Respuesta perdida DESPUÉS de un commit real en el fixture.
            page.evaluate('() => { anns[0].poligonos=[[20,25,110,25,110,90,20,90]]; manual=[]; puntos=[]; caja=null; indicacionesPorObjeto={}; dirty=true; }')
            page.check('#completa')
            def perder(route):route.fetch();route.abort('connectionfailed')
            page.route('**/api/guardar',perder);page.evaluate('() => { guardarSeguro(); guardarSeguro(); }');expect(page.locator('#mensaje')).to_contain_text('Fallo de red')
            casos.append('dos_guardados_simultaneos_bloqueados_antes_de_esperar_IndexedDB')
            assert len(t.diario('fixture:1'))==1;page.unroute('**/api/guardar',perder);page.click('#guardar')
            expect(page.locator('#mensaje')).to_contain_text('Guardado confirmado');assert len(t.diario('fixture:1'))==1
            assert page.locator('#completa').is_checked();casos.append('respuesta_perdida_reintento_idempotente_y_cobertura_preservada')
            # La nota de incidencia no debe ensuciar las anotaciones ni simular otro evento.
            page.fill('#nota_inc','Contorno y material comprobados en el fixture');page.click('#resolver_inc')
            expect(page.locator('#mensaje')).to_contain_text('Estado registrado');assert len(t.diario('fixture:1'))==1
            assert g.listar()[0]['estado']=='corregida';casos.append('resolucion_explicita_vinculada_sin_crear_otro_revisor')
            # Conservar también un polígono que todavía no se ha terminado.
            page.select_option('#imagen','fixture:2');expect(page.locator('#titulo')).to_contain_text('fixture:2')
            page.evaluate('() => { manual=[[30,30],[70,30]]; dirty=true; dibujar(); }');page.evaluate('() => persistir(true)')
            page.reload();expect(page.locator('#recuperar')).to_be_enabled();page.click('#recuperar')
            assert page.evaluate('() => manual.length')==2;assert not t.diario('fixture:2');casos.append('poligono_sin_terminar_sobrevive_recarga')
            page.evaluate('() => { manual=[]; indicacionesPorObjeto={}; dirty=true; config.token="caducado_fixture"; }')
            page.fill('#persona','actor_fixture_browser');page.click('#guardar');expect(page.locator('#error_detalle')).to_contain_text('Token local caducado')
            page.click('#contexto');expect(page.locator('#mensaje')).to_contain_text('Contexto actualizado')
            assert not t.diario('fixture:2');casos.append('token_caducado_recupera_contexto_sin_perder_borrador')
            # Fallo previo al commit, con descarga local aún disponible sin red.
            page.evaluate('() => { manual=[]; indicacionesPorObjeto={}; dirty=true; }');page.fill('#persona','actor_fixture_browser')
            contexto.set_offline(True);page.click('#guardar');expect(page.locator('#mensaje')).to_contain_text('Fallo de red')
            with page.expect_download() as dl:page.click('#emergencia')
            dl.value.save_as(r/'offline.json');assert leer(r/'offline.json')['estado']['imagen_id']=='fixture:2'
            assert not t.diario('fixture:2');contexto.set_offline(False);casos.append('fallo_red_previo_no_pierde_borrador_y_descarga_offline')
            otra=contexto.new_page();otra.goto(url+'/?imagen=fixture%3A2');expect(otra.locator('#objetos option')).to_have_count(1)
            otra.fill('#persona','otro_actor_fixture');otra.select_option('#material','papel');otra.check('#completa');otra.click('#guardar')
            expect(otra.locator('#mensaje')).to_contain_text('Guardado confirmado');antes_local=page.evaluate('() => JSON.stringify(anns)')
            page.click('#guardar');expect(page.locator('#error_detalle')).to_contain_text('Conflicto')
            with page.expect_download() as dl:page.click('#contexto')
            dl.value.save_as(r/'comparacion_conflicto.json');expect(page.locator('#mensaje')).to_contain_text('No se rebasó')
            assert page.evaluate('() => JSON.stringify(anns)')==antes_local;assert len(t.diario('fixture:2'))==1
            casos.append('dos_pestanas_conservan_borrador_y_no_rebasan_revision_ajena');otra.close()
            page.close()
        finally:a.shutdown();a.server_close();h.join();contexto.close()
        real=Taller(PROYECTO,RAIZ);antes={p.relative_to(PROYECTO).as_posix():sha(p) for p in (PROYECTO/'revisiones').rglob('*.json')}
        gr=gestion(real);app,hr,u=iniciar(real,gr,MotorFixture());ctx=b.new_context(viewport={'width':1650,'height':1100});pg=ctx.new_page();pg.on('pageerror',lambda e:errores.append(str(e)))
        try:
            pg.goto(u);expect(pg.locator('#imagen option')).to_have_count(72);expect(pg.locator('#titulo')).to_contain_text('hpi:0')
            expect(pg.locator('#persona')).to_have_value('juan_miguel');expect(pg.locator('#observacion')).to_contain_text('hpi:0')
            assert not pg.evaluate('() => dirty');pg.select_option('#filtro_prioridad','2');expect(pg.locator('#titulo')).to_contain_text('warp:')
            captura=RAIZ/'datos/nuevos/v28_3/capturas_privadas'/f'editor_{marca}.png';captura.parent.mkdir(parents=True,exist_ok=True);pg.screenshot(path=str(captura),full_page=True)
            casos.append('72_fotos_reales_navegacion_solo_lectura_sin_guardar')
        finally:app.shutdown();app.server_close();hr.join();ctx.close();b.close()
        despues={p.relative_to(PROYECTO).as_posix():sha(p) for p in (PROYECTO/'revisiones').rglob('*.json')}
        if antes!=despues or errores:raise AssertionError({'revisiones_sin_cambio':antes==despues,'errores_js':errores})
        publicar(SALIDA/'pruebas_navegador'/f'{marca}.json',{'aprobada':True,'casos':casos,'errores_js':errores,
            'revisiones_reales_sin_cambio':True,'eventos_reales':len(antes),'captura_privada':captura.relative_to(RAIZ).as_posix(),
            'GPU':False,'escrituras_solo_fixtures_descartados':True})
        print('Navegador V28.3 aprobado:',len(casos),'casos; 134 eventos reales intactos',flush=True)


if __name__=='__main__':main()
