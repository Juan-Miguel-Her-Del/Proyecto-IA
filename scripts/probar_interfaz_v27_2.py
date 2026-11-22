"""Prueba visual real: correcciones SOLO en fixture borrado; piloto de 72 solo lectura."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import uuid
import re
import argparse

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from playwright.sync_api import sync_playwright, expect
from ecowaste.incorporacion import publicar,sha
from ecowaste.auxiliares_publicos import Taller
from ecowaste.asistente_sam2 import AsistenteSAM2
from pruebas.fixtures_anotacion_v27_2 import fixture
from scripts.anotar_v27_2 import servidor,PROYECTO
from scripts.fuentes_v27_2 import SALIDA,DATOS


def arrancar(taller,motor):
    app=servidor(taller,motor,0)
    hilo=threading.Thread(target=app.serve_forever,daemon=True);hilo.start()
    return app,hilo,f'http://127.0.0.1:{app.server_port}'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--revision',type=int,default=1);args=ap.parse_args()
    if not 1<=args.revision<=99:raise ValueError('Revisión fuera de rango')
    archivo=SALIDA/('prueba_interfaz'+(f'_revision{args.revision:02d}' if args.revision>1 else '')+'.json')
    if archivo.exists():raise FileExistsError('Prueba publicada; conservarla')
    errores=[];motor=AsistenteSAM2(RAIZ);run=uuid.uuid4().hex[:10]
    real=Taller(PROYECTO,RAIZ)
    antes=list(PROYECTO.rglob('revisiones/*.json'))
    # El test no registra revisiones ficticias sobre fotografías reales.
    with tempfile.TemporaryDirectory(prefix='ecowaste_ui_fixture_') as td, sync_playwright() as p:
        root=Path(td);t=fixture(root);app,hilo,url=arrancar(t,motor)
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
        try:
            page=browser.new_page(viewport={'width':1500,'height':1080},device_scale_factor=1)
            page.on('pageerror',lambda e:errores.append(str(e)))
            page.on('dialog',lambda d:d.accept())
            page.goto(url);expect(page.locator('#objetos option')).to_have_count(1)
            page.fill('#persona','actor_fixture_browser_1');page.select_option('#material','plastico')
            page.check('#completa');page.click('#guardar')
            expect(page.locator('#estado')).to_contain_text('corregida_usuario_unico')
            assert t.diario('fixture:1')[0]['anotaciones'][0]['material']=='plastico'
            page.click('#independiente');expect(page.locator('#estado')).to_contain_text('REVISIÓN INDEPENDIENTE')
            assert page.locator('#objetos option').count()==0
            assert page.locator('#material').input_value()=='sin_asignar'
            page.check('#completa');page.click('#guardar')
            expect(page.locator('#mensaje')).to_contain_text('otra persona')
            assert len(t.diario('fixture:1'))==1
            page.fill('#persona','actor_fixture_browser_2');page.click('#nuevo');page.select_option('#material','carton')
            rect=page.locator('#lienzo').bounding_box()
            page.mouse.move(rect['x']+18,rect['y']+22);page.mouse.down()
            page.mouse.move(rect['x']+115,rect['y']+95,steps=5);page.mouse.up()
            page.click('#sam');expect(page.locator('#mensaje')).to_have_text(re.compile('^Propuesta SAM'),timeout=45000)
            page.check('#completa');page.click('#guardar')
            expect(page.locator('#estado')).to_contain_text('con_revision_independiente')
            assert len(t.diario('fixture:1'))==2
            assert not t.exportar_coco()['annotations'][0]['eco']['aceptada_V26']
            with page.expect_download() as download:
                page.click('#exportar')
            out=root/'export_fixture.json';download.value.save_as(out)
            coco=json.loads(out.read_bytes());assert not coco['info']['entrenamiento_autorizado']
            page.click('#independiente');expect(page.locator('#estado')).not_to_contain_text('REVISIÓN INDEPENDIENTE')
            page.select_option('#modo','editar');old=page.evaluate('() => anns[0].poligonos[0].slice(0,2)')
            rect=page.locator('#lienzo').bounding_box()
            page.mouse.move(rect['x']+old[0],rect['y']+old[1]);page.mouse.down()
            page.mouse.move(rect['x']+old[0]+3,rect['y']+old[1]+2,steps=4);page.mouse.up()
            assert page.evaluate('() => anns[0].poligonos[0].slice(0,2)')!=old
            page.click('#deshacer');assert page.evaluate('() => anns[0].poligonos[0].slice(0,2)')==old
            page.close()
        finally:
            app.shutdown();app.server_close();hilo.join()
        app,hilo,url=arrancar(real,motor)
        try:
            page=browser.new_page(viewport={'width':1500,'height':1080},device_scale_factor=1)
            page.on('pageerror',lambda e:errores.append(str(e)))
            page.goto(url);expect(page.locator('#imagen option')).to_have_count(72)
            expect(page.locator('#mensaje')).to_have_text(re.compile('^Selecciona'))
            assert int(page.locator('#lienzo').get_attribute('width'))>240
            assert not page.locator('#persona').input_value()
            assert 'No aceptada V26' in page.locator('#estado').inner_text()
            screenshot=DATOS/'contactos'/f'editor_real_pendiente_{run}.png'
            page.screenshot(path=str(screenshot),full_page=True)
            page.select_option('#fuente','wsc');expect(page.locator('#titulo')).to_have_text(re.compile('^wsc'))
            assert page.locator('#imagen option').count()==12
            page.select_option('#fuente','hpi');expect(page.locator('#titulo')).to_have_text(re.compile('^hpi'))
            assert page.locator('#objetos option').count()==0
            assert page.locator('#material').input_value()=='sin_asignar'
            page.close()
        finally:
            app.shutdown();app.server_close();hilo.join();browser.close();motor.liberar()
    assert not errores,errores
    assert list(PROYECTO.rglob('revisiones/*.json'))==antes
    protegidos=['ecowaste/auxiliares_publicos.py','ecowaste/asistente_sam2.py','scripts/anotar_v27_2.py',
        'scripts/probar_interfaz_v27_2.py','pruebas/fixtures_anotacion_v27_2.py',
        'interfaz/anotacion_v27_2/index.html','interfaz/anotacion_v27_2/app.js','interfaz/anotacion_v27_2/estilo.css']
    publicar(archivo,{'aprobada':True,'navegador':'Microsoft Edge headless instalado',
        'errores_JS':errores,'fixture_descartable_eliminado':not root.exists(),'imagenes_reales_solo_lectura':72,
        'revisiones_humanas_simuladas_en_piloto_real':0,'aceptadas_V26':0,'GPU_utilizada':False,
        'controles':['cambio_material','guardado_usuario_unico','vista_independiente_sin_propuestas',
                    'rechazo_mismo_actor','SAM_CPU_desde_caja','revision_independiente_fixture',
                    'exportacion_COCO_pendiente','arrastre_vertice','deshacer','fuentes_WSC_HPI','HPI_sin_mascara_global'],
        'screenshot_local_no_publicar':screenshot.relative_to(RAIZ).as_posix(),'screenshot_sha256':sha(screenshot),
        'hashes_codigo':{f:sha(RAIZ/f) for f in protegidos}})
    print('Interfaz aprobada. Fixture eliminado; 72 fotos reales sin revisiones simuladas.',flush=True)


if __name__=='__main__':main()
