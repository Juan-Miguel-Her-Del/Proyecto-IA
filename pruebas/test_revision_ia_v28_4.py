"""Sólo fixtures descartables: ninguna aprobación del proyecto real."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import uuid
import numpy as np
from PIL import Image
from ecowaste.auxiliares_publicos import rle_coco
from ecowaste.incorporacion import publicar,digest,leer,sha
from ecowaste.ajustes_ia_v28_3 import guardar_mask,decodificar
from ecowaste.revision_ia_v28_4 import (VERSION,RevisionIA,construir_unidades,validar_anotacion,qa_rle)
from scripts.anotar_v28_4 import servidor


def crear_fixture(raiz):
    raiz=Path(raiz);imagen=raiz/'foto.png';Image.new('RGB',(64,48),(90,100,110)).save(imagen)
    im={'id':'fixture:1','width':64,'height':48,'fuente':'fixture','archivo':'foto.png','sha256_archivo':sha(imagen),
        'licencia':'fixture propia descartable','objeto_fisico_id':None}
    def obj(i,m,mat='plastico'):
        return guardar_mask({'id':i,'material':mat,'descartada':False,'motivo':'Fixture de prueba',
            'poligonos':[],'huecos':[],'aceptada_V26':False},m)
    a=np.zeros((48,64),bool);a[4:22,4:22]=1
    b=np.zeros_like(a);b[14:34,14:34]=1;s=a&b
    p=np.zeros_like(a);p[5:18,42:60]=1;p[8:12,46:51]=0
    antes=[obj('a',a),obj('b',b)]
    despues=[obj('a',a&~s),obj('b',b&~s),obj('ign',s,'ambiguo'),obj('nuevo',p,'papel')]
    unidades=construir_unidades(antes,despues,im,True)
    im2={**im,'id':'fixture:2'};im3={**im,'id':'fixture:3','fuente':'hpi'}
    row=lambda image,base,units,complete,heredada:{'imagen':image,'base':base,'base_completa':complete,
        'revision_humana_base':'base_fixture','unidades':units,'advertencias':[{'motivo':'Revisar comida pequeña y solapes','objetos':[]}],
        'fuente_sha256':'fuente_fixture_IA','candidata_heredada_V283':heredada}
    catalogo={'version':VERSION,'version_IA_origen':'fixture_IA','publicacion_IA_sha256':'hash_IA_fixture',
        'archivos_fuente':{},'personas_historicas':['usuario_fixture'],
        'imagenes':[row(im,antes,unidades,True,False),row(im2,[obj('base',p,'papel')],
            construir_unidades([obj('base',p,'papel')],[obj('base',p,'papel')],im2,False),False,False),
            row(im3,[obj('organico',p,'organico')],[],True,True)]}
    catalogo['huella']=digest(catalogo);publicar(raiz/'proyecto/catalogo.json',catalogo)
    return RevisionIA(raiz/'proyecto',raiz)


class RevisionRLE(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.raiz=Path(self.tmp.name)
        self.rev=crear_fixture(self.raiz);self.iid='fixture:1'
    def request(self,accion='aceptar',uid=None,iid=None):
        iid=iid or self.iid;s=self.rev.estado(iid);units=[uid] if uid else [u['id'] for u in s['unidades']]
        return {'proyecto_sha256':self.rev.id,'imagen_id':iid,'persona_id':'usuario_fixture','accion':accion,
            'id_solicitud':'sol_'+uuid.uuid4().hex,'estado_sha256':s['estado_sha256'],'unidades':units,
            'inspeccionadas':units,'huellas_unidades':{u['id']:u['huella'] for u in s['unidades']},
            'nota':'Decisión explícita de un usuario de fixture'}
    def confirmar(self,iid=None):
        iid=iid or self.iid;t=self.rev.tarea(iid);d=self.request('confirmar_cobertura',iid=iid)
        d.update(confirmo_cobertura=True,advertencias_inspeccionadas=[a['id'] for a in t['advertencias']])
        return self.rev.guardar(d)
    def test_cambios_dependientes_son_atomicos(self):
        us=self.rev.estado(self.iid)['unidades'];self.assertEqual(len(us),2)
        self.assertEqual(sorted(len(u['objetos']) for u in us),[1,3])
    def test_no_aprueba_por_abrir_ni_por_aceptar(self):
        self.rev.tarea(self.iid);self.assertEqual(self.rev.eventos(),[])
        self.rev.guardar(self.request());self.assertFalse(self.rev.estado(self.iid)['cobertura_completa'])
    def test_no_acepta_media_unidad(self):
        d=self.request();d['unidades']=['a']
        with self.assertRaises(ValueError):self.rev.guardar(d)
        self.assertEqual(self.rev.eventos(),[])
    def test_inspeccion_explicita_obligatoria(self):
        d=self.request();d['inspeccionadas']=[]
        with self.assertRaises(ValueError):self.rev.guardar(d)
    def test_aceptacion_registra_persona_fuente_y_hash(self):
        e=self.rev.guardar(self.request())['evento']
        self.assertEqual(e['persona_id'],'usuario_fixture');self.assertEqual(e['publicacion_IA_sha256'],'hash_IA_fixture')
        self.assertEqual(e['huella'],digest({k:v for k,v in e.items() if k!='huella'}))
        self.assertTrue(e['revision_humana_individual']);self.assertFalse(e['revision_independiente']);self.assertFalse(e['aceptada_V26'])
    def test_rechazo_restaura_base_sin_borrar_fuente(self):
        self.rev.guardar(self.request('rechazar'));s=self.rev.estado(self.iid)
        self.assertEqual({a['id'] for a in s['anotaciones']},{'a','b'})
        self.assertEqual(len(self.rev.imagenes[self.iid]['unidades']),2)
    def test_posponer_no_resuelve(self):
        self.rev.guardar(self.request('posponer'));self.assertEqual(self.rev.estado(self.iid)['pendientes'],2)
    def test_cobertura_requiere_todas_las_decisiones(self):
        with self.assertRaises(ValueError):self.confirmar()
    def test_cobertura_es_otra_decision_y_se_invalida(self):
        self.rev.guardar(self.request());self.confirmar();self.assertTrue(self.rev.estado(self.iid)['cobertura_completa'])
        self.rev.guardar(self.request('posponer'));self.assertFalse(self.rev.estado(self.iid)['cobertura_completa'])
    def test_cobertura_exige_advertencias_y_casilla(self):
        self.rev.guardar(self.request());d=self.request('confirmar_cobertura');d['confirmo_cobertura']=False
        with self.assertRaises(ValueError):self.rev.guardar(d)
        d['confirmo_cobertura']=True;d['advertencias_inspeccionadas']=[]
        with self.assertRaises(ValueError):self.rev.guardar(d)
    def test_rechazar_base_no_revisada_no_la_aprueba(self):
        self.rev.guardar(self.request('rechazar',iid='fixture:2'))
        with self.assertRaises(ValueError):self.confirmar('fixture:2')
    def test_conflicto_no_sobrescribe(self):
        d=self.request();otro=self.request('rechazar');self.rev.guardar(d)
        with self.assertRaises(ValueError):self.rev.guardar(otro)
        self.assertEqual(len(self.rev.eventos()),1)
    def test_reintento_despues_de_reiniciar_no_duplica(self):
        d=self.request();self.rev.guardar(d);self.rev=RevisionIA(self.rev.path,self.raiz)
        d['token_vista']='sesion_nueva';d['tokens_inspeccion']={'nueva':'sesion'}
        self.assertTrue(self.rev.guardar(d)['reintento_idempotente']);self.assertEqual(len(self.rev.eventos()),1)
    def test_mismo_id_distinto_contenido_rechazado(self):
        d=self.request();self.rev.guardar(d);d['nota']='Otra nota cambia la decisión'
        with self.assertRaises(ValueError):self.rev.guardar(d)
    def test_fuente_humana_nueva_bloquea_decision(self):
        class TallerCambiado:
            def diario(self,iid):return [{'modo':'asistida','huella':'otra_revision'}]
        self.rev.taller=TallerCambiado()
        with self.assertRaises(ValueError):self.rev.guardar(self.request())
    def test_fuente_IA_alterada_despues_de_abrir_bloquea(self):
        p=self.raiz/'fuente.json';p.write_text('{}',encoding='utf8')
        c=leer(self.rev.path/'catalogo.json');c['archivos_fuente']={'fuente.json':sha(p)}
        c['huella']=digest({k:v for k,v in c.items() if k!='huella'})
        (self.rev.path/'catalogo.json').write_text(json.dumps(c),encoding='utf8')
        self.rev=RevisionIA(self.rev.path,self.raiz);p.write_text('{"modificada":true}',encoding='utf8')
        with self.assertRaises(ValueError):self.rev.guardar(self.request())
        self.assertEqual(self.rev.eventos(),[])
    def test_catalogo_alterado_despues_de_abrir_bloquea(self):
        p=self.rev.path/'catalogo.json';p.write_text(p.read_text(encoding='utf8')+' ',encoding='utf8')
        with self.assertRaises(ValueError):self.rev.guardar(self.request())
    def test_cadena_alterada_detectada(self):
        self.rev.guardar(self.request());p=self.rev.path/'eventos/000001.json';e=leer(p);e['persona_id']='impostor';p.write_text(json.dumps(e),encoding='utf8')
        with self.assertRaises(ValueError):self.rev.eventos()
    def test_edicion_no_puede_borrar_ids_sin_descarte(self):
        u=max(self.rev.estado(self.iid)['unidades'],key=lambda x:len(x['objetos']));d=self.request('editar',u['id']);d['anotaciones_editadas']=u['propuesta'][:1]
        with self.assertRaises(ValueError):self.rev.guardar(d)
    def test_edicion_rle_material_y_hueco_preservados(self):
        u=next(u for u in self.rev.estado(self.iid)['unidades'] if u['objetos']==['nuevo']);d=self.request('editar',u['id']);anns=deepcopy(u['propuesta']);anns[0]['material']='carton';d['anotaciones_editadas']=anns
        self.rev.guardar(d);a=next(a for a in self.rev.estado(self.iid)['anotaciones'] if a['id']=='nuevo')
        self.assertEqual(a['material'],'carton');self.assertEqual(a['segmentacion_canonica_RLE'],anns[0]['segmentacion_canonica_RLE'])
        self.assertTrue(a['revision_V284']['ajuste_IA_aceptado'])
    def test_rle_malformado_o_de_otras_dimensiones_no_se_guarda(self):
        im=self.rev.imagenes[self.iid]['imagen'];a=deepcopy(self.rev.imagenes[self.iid]['base'][0])
        for r in [{'size':[48,64],'counts':[True,3071]},{'size':[48,64],'counts':[3071]},{'size':[4,4],'counts':[16]}]:
            a['segmentacion_canonica_RLE']=r
            with self.assertRaises(ValueError):validar_anotacion(a,im)
    def test_no_se_acepta_mascara_vacia(self):
        a=deepcopy(self.rev.imagenes[self.iid]['base'][0]);a['segmentacion_canonica_RLE']={'size':[48,64],'counts':[3072]}
        with self.assertRaises(ValueError):validar_anotacion(a,self.rev.imagenes[self.iid]['imagen'])
    def test_ignore_requiere_motivo_y_sin_asignar_bloquea(self):
        a=deepcopy(self.rev.imagenes[self.iid]['base'][0]);a.update(material='ambiguo',motivo='')
        with self.assertRaises(ValueError):validar_anotacion(a,self.rev.imagenes[self.iid]['imagen'])
        a.update(material='sin_asignar',motivo='Debe resolver usuario')
        with self.assertRaises(ValueError):validar_anotacion(a,self.rev.imagenes[self.iid]['imagen'])
    def test_nuevo_objeto_no_puede_usurpar_id_ajeno(self):
        u=next(u for u in self.rev.estado(self.iid)['unidades'] if u['objetos']==['nuevo']);d=self.request('editar',u['id']);d['anotaciones_editadas']=[*deepcopy(u['propuesta']),deepcopy(self.rev.imagenes[self.iid]['base'][0])]
        with self.assertRaises(ValueError):self.rev.guardar(d)
    def test_objeto_anadido_por_usuario_puede_editarse_otra_vez(self):
        u=next(u for u in self.rev.estado(self.iid)['unidades'] if u['objetos']==['nuevo'])
        anns=deepcopy(u['propuesta']);nuevo=deepcopy(anns[0]);nuevo['id']='usuario_v284_objeto';nuevo['material']='papel'
        m=decodificar(nuevo['segmentacion_canonica_RLE']);nuevo['segmentacion_canonica_RLE']=rle_coco(np.roll(m,22,axis=0))
        d=self.request('editar',u['id']);d['anotaciones_editadas']=[*anns,nuevo];self.rev.guardar(d)
        d=self.request('editar',u['id']);d['anotaciones_editadas']=[*anns,{**nuevo,'material':'carton'}]
        self.rev.guardar(d);self.assertEqual(len(self.rev.eventos()),2)
        self.assertEqual(next(a['material'] for a in self.rev.estado(self.iid)['anotaciones'] if a['id']==nuevo['id']),'carton')
    def test_confirmados_por_usuario_separados_de_utilizables(self):
        self.rev.guardar(self.request());_,s=self.rev.auditar()
        self.assertEqual(s['positivos_confirmados_en_propuestas_V284'],3)
        self.assertEqual(s['ignore_confirmados_en_propuestas_V284'],1)
        self.assertEqual(s['imagenes_nuevas_confirmadas'],0);self.assertEqual(s['objetos_utilizables'],1)
    def test_referencia_sam_inventada_no_se_acepta(self):
        u=next(u for u in self.rev.estado(self.iid)['unidades'] if u['objetos']==['nuevo']);d=self.request('editar',u['id']);d['anotaciones_editadas']=deepcopy(u['propuesta']);d['anotaciones_editadas'][0]['sam_v284_id']='sam_inexistente'
        with self.assertRaises(OSError):self.rev.guardar(d)
        self.assertEqual(self.rev.eventos(),[])
    def test_auditoria_cache_invalida_solo_imagen_modificada(self):
        self.rev.auditar();_,s=self.rev.auditar();self.assertEqual(s['cache_reusadas'],3)
        self.rev.guardar(self.request());_,s=self.rev.auditar();self.assertEqual(s['cache_reusadas'],2)
        self.assertEqual(s['imagenes_nuevas_confirmadas'],0)
    def test_contador_imagenes_no_sube_hasta_cobertura(self):
        _,s=self.rev.auditar();self.assertEqual(s['imagenes_utilizables'],1)
        self.rev.guardar(self.request());self.confirmar();_,s=self.rev.auditar()
        self.assertEqual(s['imagenes_utilizables'],2);self.assertEqual(s['imagenes_nuevas_confirmadas'],1)
    def test_auditoria_queda_desactualizada_tras_decision(self):
        self.rev.auditar();self.assertTrue(self.rev.ultima_auditoria()['vigente']);self.rev.guardar(self.request())
        self.assertFalse(self.rev.ultima_auditoria()['vigente'])
        with self.assertRaises(ValueError):self.rev.exportar()
    def test_cache_y_cadena_auditoria_alteradas_se_rechazan(self):
        p,_=self.rev.auditar();f=p.parent/'imagenes.json';f.write_text('[]',encoding='utf8')
        with self.assertRaises(ValueError):self.rev.ultima_auditoria()
    def test_exportador_conserva_huecos_ignore_y_procedencia(self):
        self.rev.guardar(self.request());self.confirmar();self.rev.auditar();_,c=self.rev.exportar()
        m={a['eco_objeto_id']:a for a in c['annotations'] if a['image_id']==next(i['id'] for i in c['images'] if i['eco_imagen_id']==self.iid)}
        self.assertEqual(m['ign']['ignore'],1);self.assertEqual(m['nuevo']['ignore'],0)
        mask=decodificar(m['nuevo']['segmentation']);self.assertFalse(mask[9,48]);self.assertTrue(mask[6,44])
        self.assertFalse(c['info']['aceptadas_V26']);self.assertEqual(c['info']['estado'],'NO_ENTRENAR_AUXILIAR_USUARIO_UNICO')
        self.assertTrue(m['nuevo']['eco']['revision_V284']['ajuste_IA_aceptado'])
        self.assertFalse(m['nuevo']['eco']['base_humana_anterior'])
        self.assertTrue(m['a']['eco']['base_humana_anterior'])
    def test_exportacion_pendiente_no_se_declara_aprobada(self):
        self.rev.auditar();_,c=self.rev.exportar(False)
        self.assertFalse(c['info']['solo_utilizables']);self.assertFalse(c['info']['aceptadas_V26'])
        self.assertFalse(next(i for i in c['images'] if i['eco_imagen_id']==self.iid)['cobertura_humana_V284'])
    def test_exportacion_no_sobrescribe_historia(self):
        self.rev.auditar();p,_=self.rev.exportar();h=sha(p/'annotations.json');p2,_=self.rev.exportar()
        self.assertNotEqual(p,p2);self.assertEqual(h,sha(p/'annotations.json'))
    def test_rejilla_excluye_ignore_padding_y_mezclas(self):
        r=self.rev.imagenes[self.iid];q=qa_rle(r['imagen'],r['unidades'][0]['propuesta'])
        self.assertGreater(q['tokens_ignore'],0)
        self.assertTrue(all(a['componente_maximo_48']>=0 for a in q['objetos']))


class HTTPRevision(RevisionRLE):
    # No heredar los tests de dominio dos veces.
    def setUp(self):
        super().setUp();self.app=servidor(self.rev);self.hilo=threading.Thread(target=self.app.serve_forever,daemon=True);self.hilo.start();self.url=f'http://127.0.0.1:{self.app.server_port}'
        self.addCleanup(self.cerrar);self.cfg=self.get('/api/config')
    def cerrar(self):
        self.app.shutdown();self.app.detener_auditor();self.app.server_close();self.hilo.join(timeout=5)
    def get(self,path):
        with urllib.request.urlopen(self.url+path,timeout=20) as r:return json.load(r)
    def post(self,path,d,token=True):
        h={'Content-Type':'application/json'}
        if token:h['X-Eco-Token']=self.cfg['token']
        req=urllib.request.Request(self.url+path,data=json.dumps(d).encode(),headers=h)
        with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
    def real_request(self):
        t=self.get('/api/tarea?id=fixture%3A1');d=self.request();d['token_vista']=t['token_vista'];d['tokens_inspeccion']={}
        for u in t['unidades']:d['tokens_inspeccion'][u['id']]=self.get('/api/inspeccion?id=fixture%3A1&unidad='+u['id'])['token_inspeccion']
        return d
    def test_http_sin_token_o_sin_inspeccion_no_crea_evento(self):
        with self.assertRaises(urllib.error.HTTPError):self.post('/api/guardar',self.request(),False)
        d=self.real_request();d['tokens_inspeccion']={}
        with self.assertRaises(urllib.error.HTTPError):self.post('/api/guardar',d)
        self.assertEqual(self.rev.eventos(),[]);self.assertTrue(list((self.rev.path/'borradores_servidor').glob('*.json')))
    def test_http_guardado_y_reintento_con_recibo(self):
        d=self.real_request();r=self.post('/api/guardar',d);self.assertEqual(r['evento']['persona_id'],'usuario_fixture')
        self.assertTrue(self.post('/api/guardar',d)['reintento_idempotente']);self.assertEqual(len(self.rev.eventos()),1)
        self.assertTrue(self.get('/api/recibo?id='+d['id_solicitud'])['encontrado'])
    def test_http_no_hay_aprobacion_global_por_imagen(self):
        with self.assertRaises(urllib.error.HTTPError):self.post('/api/aprobar_todas',{})
        self.assertEqual(self.rev.eventos(),[])


# Separar la clase HTTP sin duplicar pruebas heredadas en discover.
for _nombre in [n for n in RevisionRLE.__dict__ if n.startswith('test_')]:
    setattr(HTTPRevision,_nombre,None)


if __name__=='__main__':unittest.main()
