"""QA train-only, respaldo e impedimento de entrenamiento; fixtures descartables."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ecowaste.validacion_anotaciones import (analizar_imagen,comparar,mask_poligonos,celdas,revision_humana_final)
from ecowaste.preparacion_v28 import suficiencia,exigir_ejecucion
from ecowaste.incorporacion import CLASES
from scripts.respaldar_anotaciones_v28 import respaldar,inventariar


def objeto(aid='a',material='organico',x=40,y=40,w=100,h=100):
    return {'id':aid,'material':material,'poligonos':[[x,y,x+w,y,x+w,y+h,x,y+h]],'huecos':[],
            'bbox':[x,y,w,h],'motivo':'','descartada':False,'aceptada_V26':False,'origen':'fixture'}


IM={'id':'fixture','width':384,'height':384}


class ValidacionV28(unittest.TestCase):
    def test_respaldo_todos_bytes_y_lock_sin_alterar_origen(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);o=r/'original';o.mkdir();(o/'registro.lock').write_bytes(b'0')
            (o/'revisiones').mkdir();(o/'revisiones/1.json').write_bytes(b'{"no_es_revision_real":true}')
            (o/'propuesta.bin').write_bytes(b'original\x00')
            antes=inventariar(o);h=respaldar(o,r/'copia')
            self.assertEqual(h,antes);self.assertEqual(inventariar(o),antes)
            with self.assertRaises(ValueError):respaldar(o,r/'copia')

    def test_respaldo_rechaza_destino_dentro_del_original(self):
        with tempfile.TemporaryDirectory() as td:
            o=Path(td)
            with self.assertRaises(ValueError):respaldar(o,o/'hijo')

    def test_sin_evento_no_hay_anotaciones_humanas(self):
        self.assertIsNone(revision_humana_final([]))
        r=analizar_imagen(IM,[],False);self.assertEqual(r['objetos'],[]);self.assertEqual(r['organicos_utilizables_48'],0)

    def test_borrador_mas_reciente_no_reutiliza_cobertura_antigua(self):
        es=[{'modo':'asistida','cobertura_completa':True},{'modo':'asistida','cobertura_completa':False},
            {'modo':'independiente','cobertura_completa':True}]
        self.assertFalse(revision_humana_final(es)['cobertura_completa'])

    def test_confirmar_una_propuesta_no_es_cambiar_su_geometria(self):
        a=objeto();d=comparar([a],[deepcopy(a)],IM)
        self.assertEqual(d['contornos_cambiados'],[]);self.assertEqual(d['agregados'],[])
        self.assertEqual(d['sin_cambio_coordenadas_material'],['a'])

    def test_deltas_anadidos_reclasificados_y_descartados_sin_doble_conteo(self):
        a=objeto(material='plastico');b={**deepcopy(a),'material':'papel','descartada':True,'motivo':'propuesta falsa'}
        d=comparar([a],[b,objeto('nuevo')],IM)
        self.assertEqual(d['agregados'],['nuevo']);self.assertEqual(d['descartados'],['a'])
        self.assertEqual(d['reclasificados'][0]['antes'],'plastico')

    def test_borrado_de_original_se_detecta(self):
        d=comparar([objeto()],[],IM)
        self.assertEqual(d['originales_ausentes_sin_registro'],['a'])

    def test_geometria_invalida_se_informa_sin_repararla(self):
        a=objeto();a['poligonos']=[[1,1,50,50,1,50,50,1]];antes=deepcopy(a)
        r=analizar_imagen(IM,[a],True)
        self.assertTrue(r['errores']);self.assertEqual(a,antes)

    def test_cierre_ciclico_y_caja_derivada(self):
        a=objeto();a['poligonos'][0]+=[40,40];antes=deepcopy(a)
        r=analizar_imagen(IM,[a],True)
        self.assertFalse(r['errores']);self.assertEqual(r['objetos'][0]['bbox_derivada'],[40.,40.,100.,100.])
        self.assertEqual(a,antes)

    def test_huecos_reales_se_excluyen_de_la_mascara(self):
        a=objeto();a['huecos']=[[60,60,80,60,80,80,60,80]]
        m=mask_poligonos(a,384,384);self.assertEqual(m[70,70],0);self.assertEqual(m[50,50],1)

    def test_diferencia_union_vs_editor_se_preserva_como_incidencia(self):
        a=objeto();a['poligonos'].append([100,100,160,100,160,160,100,160]);a['bbox']=[40,40,120,120]
        r=analizar_imagen(IM,[a],True)
        self.assertIn('relleno_paridad_editor_difiere_union_exportador',r['objetos'][0]['errores'])
        self.assertGreater(r['objetos'][0]['pixeles_discrepancia_editor_exportador'],0)

    def test_padding_no_produce_organicos_utilizables(self):
        im={**IM,'width':384,'height':32};a=objeto(x=3,y=2,w=1,h=1)
        r=analizar_imagen(im,[a],True)
        self.assertGreater(r['tokens_padding_excluidos'],0);self.assertEqual(r['organicos_utilizables_48'],0)

    def test_organico_util_requiere_dos_tokens_y_cobertura(self):
        self.assertEqual(analizar_imagen(IM,[objeto()],True)['organicos_utilizables_48'],1)
        self.assertEqual(analizar_imagen(IM,[objeto()],False)['organicos_utilizables_48'],0)
        self.assertEqual(analizar_imagen(IM,[objeto(w=1,h=1)],True)['organicos_utilizables_48'],0)

    def test_mezcla_de_material_en_token_no_acredita_region(self):
        a=objeto(x=1,y=1,w=2,h=2);b=objeto('b','plastico',x=4,y=1,w=2,h=2)
        r=analizar_imagen(IM,[a,b],True)
        self.assertGreater(r['tokens_mezcla_material'],0);self.assertEqual(r['objetos'][0]['tokens_regionales'],0)

    def test_compuesto_se_preserva_como_ignore(self):
        a=objeto();b=objeto('b','compuesto');b['motivo']='Envase multicapa'
        r=analizar_imagen(IM,[a,b],True);self.assertEqual(r['organicos_utilizables_48'],0)
        self.assertEqual(len(r['objetos']),2)

    def test_fusion_en_rejilla_no_fusiona_las_instancias_GT(self):
        a=objeto(x=0,y=0,w=15,h=15);b=objeto('b',x=16,y=0,w=15,h=15)
        r=analizar_imagen(IM,[a,b],True)
        self.assertTrue(r['fusiones_48']);self.assertEqual(len(r['objetos']),2)

    def test_avisos_y_observaciones_no_equivalen_a_datos_listos(self):
        rows=[{'imagen':str(i),'estado':'completa_usuario_unico','errores':[],'avisos_visuales':[],
            'qa_objetos':[{'material':c,'descartada':False,'valido_geometria':True} for c in CLASES]} for i in range(40)]
        self.assertTrue(suficiencia(rows,[])['cumple_referencia_numerica'])
        s=suficiencia(rows,[{'imagen':'0','requiere_decision':True}])
        self.assertEqual(s['cantidad_imagenes'],39);self.assertFalse(s['cumple_referencia_numerica'])
        self.assertFalse(s['entrenamiento_autorizado'])

    def test_autorizacion_y_QA_son_barreras_independientes(self):
        with self.assertRaises(PermissionError):exigir_ejecucion({'entrenamiento_autorizado':False},{})
        with self.assertRaises(PermissionError):exigir_ejecucion({'entrenamiento_autorizado':True},{'aprobada_qa_para_entrenar':False})
        with self.assertRaises(ValueError):exigir_ejecucion({'entrenamiento_autorizado':True,'hash_datos':'otro'},
             {'aprobada_qa_para_entrenar':True,'coleccion_congelada_para_entrenar':True,'hash_datos':'este'})


if __name__=='__main__':unittest.main()
