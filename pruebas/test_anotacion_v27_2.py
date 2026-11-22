"""Contrato auxiliar y correcciones trazables, sin simular humanos en fotos reales."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ecowaste.auxiliares_publicos import Taller, normalizar, rle_coco, contornos_mask, mascara_poligonos
from ecowaste.asistente_sam2 import AsistenteSAM2
from ecowaste.incorporacion import leer, digest, publicar
from pruebas.fixtures_anotacion_v27_2 import fixture


def decodificar(rle):
    a, valor = [], 0
    for n in rle['counts']:
        a.extend([valor] * n)
        valor = 1 - valor
    return np.array(a, dtype=np.uint8).reshape(rle['size'], order='F')


class AuxiliaresV272(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ecowaste_fixture_v272_')
        self.root = Path(self.temp.name)
        self.t = fixture(self.root)
        self.iid = 'fixture:1'
        self.im = self.t.imagen(self.iid)
        self.a = deepcopy(self.im['anotaciones'][0])

    def tearDown(self):
        self.temp.cleanup()

    def guardar(self, *, actor='actor_fixture_1', completa=True, anns=None, modo='asistida', base=None):
        return self.t.guardar(self.iid, {'persona_id': actor, 'cobertura_completa': completa,
            'revision': len(self.t.diario(self.iid)), 'anotaciones': [deepcopy(self.a)] if anns is None else anns},
            modo=modo, revision_base=base)

    def test_proyecto_rechaza_otro_protocolo_y_metadatos_ficticios(self):
        for campo, valor in [('version','v26'),('entrenamiento_autorizado',True)]:
            p = deepcopy(self.t.base);p[campo] = valor;p['huella'] = digest({k:v for k,v in p.items() if k!='huella'})
            (self.t.path/'PROYECTO.json').write_text(json.dumps(p),encoding='utf-8')
            with self.assertRaises(ValueError):Taller(self.t.path,self.root)
        p = deepcopy(self.t.base);p['imagenes'][0]['objeto_fisico_id']='inventado'
        p['huella']=digest({k:v for k,v in p.items() if k!='huella'})
        (self.t.path/'PROYECTO.json').write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaises(ValueError):Taller(self.t.path,self.root)

    def test_foto_alterada_no_admite_revision(self):
        (self.root/'fixture.png').write_bytes(b'alterada')
        with self.assertRaises(ValueError):self.guardar()
        self.assertEqual(self.t.diario(self.iid),[])

    def test_ruta_no_puede_escapar(self):
        self.im['archivo']='../ajena.png'
        with self.assertRaises(ValueError):self.t.archivo_imagen(self.iid)

    def test_rle_preserva_huecos_componentes_y_orden(self):
        m=np.zeros((11,17),dtype=np.uint8);m[1:9,2:10]=1;m[3:6,4:7]=0;m[2:4,14:16]=1
        self.assertTrue(np.array_equal(m,decodificar(rle_coco(m))))
        m[0,0]=1;self.assertEqual(rle_coco(m)['counts'][0],0)
        self.assertTrue(np.array_equal(m,decodificar(rle_coco(m))))

    def test_mascara_vacia_y_no_binaria_rechazadas(self):
        with self.assertRaises(ValueError):contornos_mask(np.zeros((8,8),dtype=np.uint8))
        with self.assertRaises(ValueError):rle_coco(np.full((3,4),2))

    def test_contornos_no_inventan_ids_de_instancia(self):
        m=np.zeros((50,50),dtype=np.uint8);m[3:18,4:22]=1;m[30:43,33:44]=1
        r=contornos_mask(m)
        self.assertEqual(r['componentes_ia'],2);self.assertEqual(len(r['poligonos']),2)
        self.assertNotIn('objeto_fisico_id',r);self.assertTrue(np.array_equal(m,decodificar(r['rle_ia_original'])))

    def test_caja_sin_poligono_solo_admite_borrador(self):
        a={**self.a,'poligonos':[],'material':'sin_asignar'}
        self.guardar(anns=[a],completa=False)
        self.assertEqual(self.t.tarea(self.iid)['estado'],'borrador_usuario')
        with self.assertRaises(ValueError):self.guardar(anns=[a])

    def test_material_ignorado_y_descarte_requieren_motivo(self):
        for a in [{**self.a,'material':'ambiguo'},{**self.a,'descartada':True}]:
            with self.assertRaises(ValueError):normalizar(a,self.im,completa=True)
        a={**self.a,'material':'compuesto','motivo':'Envase multicapa visible'}
        self.assertEqual(normalizar(a,self.im,completa=True)['material'],'compuesto')

    def test_poligonos_fuera_de_limites_o_con_cruce_rechazados(self):
        for ps in [[[-1,0,20,0,20,20]],[[0,0,30,30,0,30,30,0]]]:
            with self.assertRaises(ValueError):normalizar({**self.a,'poligonos':ps},self.im,completa=True)

    def test_usuario_unico_no_equivale_a_aceptacion_V26(self):
        s=self.guardar()
        self.assertEqual(s['estado'],'corregida_usuario_unico');self.assertFalse(s['aceptada_V26'])
        e=self.t.diario(self.iid)[0];self.assertEqual(e['cuotas_identidad_acreditadas'],0)
        self.assertFalse(e['entrenamiento_autorizado'])
        self.assertTrue(all(self.t.imagen(self.iid)[k] is None for k in ['objeto_fisico_id','sesion_id','capturado_en']))

    def test_aceptacion_declarada_por_cliente_rechazada(self):
        with self.assertRaises(ValueError):self.t.guardar(self.iid,{'persona_id':'fixture','revision':0,'aceptada_V26':True,'anotaciones':[self.a]})
        with self.assertRaises(ValueError):normalizar({**self.a,'aceptada':True},self.im)

    def test_concurrencia_no_sobrescribe_revision(self):
        self.guardar();primeros=list((self.t.path/'revisiones').rglob('*.json'))
        with self.assertRaises(ValueError):self.t.guardar(self.iid,{'persona_id':'otro','revision':0,'cobertura_completa':True,'anotaciones':[self.a]})
        self.assertEqual(len(list((self.t.path/'revisiones').rglob('*.json'))),len(primeros))

    def test_reanudacion_conserva_correcciones_y_diario(self):
        self.guardar();antes=(self.t.path/'PROYECTO.json').read_bytes()
        self.t=Taller(self.t.path,self.root);self.a['material']='papel';self.guardar()
        evs=self.t.diario(self.iid);self.assertEqual(len(evs),2);self.assertEqual(evs[-1]['cambios'][0]['antes']['material'],'carton')
        self.assertEqual(evs[-1]['cambios'][0]['despues']['material'],'papel')
        self.assertEqual(antes,(self.t.path/'PROYECTO.json').read_bytes())

    def test_diario_alterado_no_se_exporta(self):
        self.guardar();f=next((self.t.path/'revisiones').rglob('*.json'));r=leer(f);r['persona_id']='suplantado'
        f.write_text(json.dumps(r),encoding='utf-8')
        with self.assertRaises(ValueError):self.t.exportar_coco()

    def test_independiente_oculta_propuestas_y_exige_otra_persona(self):
        self.guardar();blind=self.t.tarea(self.iid,independiente=True)
        self.assertEqual(blind['anotaciones'],[])
        with self.assertRaises(ValueError):self.guardar(modo='independiente',base=blind['revision_base'])
        s=self.guardar(actor='actor_fixture_2',modo='independiente',base=blind['revision_base'])
        self.assertEqual(s['estado'],'con_revision_independiente_pendiente_adjudicacion');self.assertFalse(s['aceptada_V26'])

    def test_independiente_no_puede_revisar_base_ausente_o_antigua(self):
        with self.assertRaises(ValueError):self.guardar(actor='fixture_2',modo='independiente')
        self.guardar();blind=self.t.tarea(self.iid,independiente=True);self.a['material']='papel';self.guardar()
        with self.assertRaises(ValueError):self.guardar(actor='fixture_2',modo='independiente',base=blind['revision_base'])

    def test_cambio_posterior_invalida_revision_independiente_previa(self):
        self.guardar();base=self.t.tarea(self.iid)['revision_base'];self.guardar(actor='fixture_2',modo='independiente',base=base)
        self.a['material']='papel';s=self.guardar()
        self.assertEqual(s['estado'],'corregida_usuario_unico')

    def test_no_borrar_objetos_sin_justificacion(self):
        with self.assertRaises(ValueError):self.guardar(anns=[])
        s=self.guardar(anns=[{**self.a,'descartada':True,'motivo':'No es residuo, verificar escena'}])
        self.assertTrue(s['anotaciones'][0]['descartada']);self.assertFalse(self.t.exportar_coco()['annotations'])

    def test_coco_deriva_caja_y_conserva_desconocidos(self):
        self.guardar();c=self.t.exportar_coco();a=c['annotations'][0];im=c['images'][0]
        self.assertEqual(a['bbox'],[20,25,90,65]);self.assertIsNone(im['eco']['objeto_fisico_id'])
        self.assertFalse(c['info']['entrenamiento_autorizado']);self.assertEqual(c['physical_objects'],[])
        self.assertFalse(a['eco']['aceptada_V26']);self.assertEqual(len(im['eco']['revisiones_huellas']),1)

    def test_huecos_no_se_rellenan_en_exportacion(self):
        self.a['huecos']=[[40,40,60,40,60,60,40,60]];self.guardar()
        a=self.t.exportar_coco()['annotations'][0];self.assertIsInstance(a['segmentation'],dict)
        mask=decodificar(a['segmentation']);self.assertEqual(mask[50,50],0);self.assertEqual(mask[30,30],1)
        self.assertTrue(a['eco']['requiere_revision_huecos_V26'])
        with self.assertRaises(ValueError):normalizar({**self.a,'huecos':[[150,120,160,120,160,130,150,130]]},self.im)

    def test_propuesta_ia_no_crea_revision_y_se_verifica_su_procedencia(self):
        r=self.t.registrar_ia(self.iid,{'poligonos':self.a['poligonos']},{'caja':[20,25,110,90]},{'nombre':'fixture_no_modelo_real'})
        self.assertFalse(self.t.diario(self.iid));self.assertFalse(r['aceptada_V26'])
        self.a['propuesta_ia_id']=r['id'];self.guardar()
        p=self.t.path/'propuestas_ia'/(r['id']+'.json');x=leer(p);x['resultado']['poligonos']=[];p.write_text(json.dumps(x),encoding='utf-8')
        with self.assertRaises(ValueError):self.guardar()

    def test_sam_rechaza_indicaciones_invalidas_antes_de_cargar_pesos(self):
        motor=AsistenteSAM2(self.root)
        for datos in [{},{'caja':[0,0,0,20]},{'caja':[0,0,500,20]},{'puntos':[[5,5]],'etiquetas':[]},
                      {'puntos':[[float('nan'),5]],'etiquetas':[1]},{'puntos':[[5,5]],'etiquetas':[2]}]:
            with self.subTest(datos=datos),self.assertRaises(ValueError):motor.proponer(self.t,self.iid,datos)
        self.assertIsNone(motor.predictor)

    def test_indice_ia_no_puede_mezclar_imagenes(self):
        r=self.t.registrar_ia(self.iid,{'poligonos':self.a['poligonos'],'huecos':[]},{'caja':[20,25,110,90]},{'nombre':'fixture'})
        p=self.t.path/'propuestas_ia'/(r['id']+'.json');r['imagen_id']='otra_imagen'
        r['huella']=digest({k:v for k,v in r.items() if k!='huella'});p.write_text(json.dumps(r),encoding='utf-8')
        publicar(self.t.path/'PROPUESTAS_IA_INICIALES.json',{'proyecto_sha256':self.t.base['huella'],
                 'por_imagen':{self.iid:{self.a['id']:r['id']}}})
        with self.assertRaises(ValueError):self.t.tarea(self.iid)


if __name__=='__main__':unittest.main()
