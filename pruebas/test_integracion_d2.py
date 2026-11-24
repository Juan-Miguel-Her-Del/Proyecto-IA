"""Regresión geométrica, taxonomía y aplicación; los mocks no son evidencia de precisión."""
import json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np,torch
from ecowaste.integracion_d2 import MotorD2,BancoClasificadores,leer_config,permutacion,preparar_transformer,verificar_pesos,RAIZ
from ecowaste.vision_equipo.detector import VisionLocaliza,decode_predictions
from ecowaste.vision_equipo.localizador import NeuralDetector,Detection
from ecowaste.vision_equipo.preprocesado import crop_box,Region,letterbox,image_tensor
from ecowaste.vision_equipo.metricas import fuse_detection_views
from ecowaste.aplicacion_integrada import video,fotografia,proteger_imagen,crear_motor,dibujar
class IntegracionD2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_arquitectura_propia_630370_parametros(self):
        m=VisionLocaliza().eval();self.assertEqual(sum(p.numel() for p in m.parameters()),630370)
        with torch.inference_mode():r=m(torch.zeros(1,3,128,128))
        self.assertEqual([tuple(t.shape) for t in r],[(1,6,16,16),(1,6,8,8),(1,6,4,4)])
    def test_organicos_reordenamiento_explicito(self):
        self.assertEqual(permutacion(['organicos','metal','vidrio','carton','papel','plastico']),[5,4,3,2,1,0])
    def test_materiales_duplicados_no_se_mezclan(self):
        with self.assertRaises(ValueError):permutacion(['plastico']*6)
    def test_material_ajeno_rechazado(self):
        with self.assertRaises(ValueError):permutacion(['plastico','papel','carton','vidrio','metal','vegetacion_viva'])
    def test_transformer_rgb224_sin_normalizar_dos_veces(self):
        a=np.zeros((40,80,3),np.uint8);a[:,:,0]=255;x=preparar_transformer(a)
        self.assertEqual(tuple(x.shape),(3,224,224));self.assertEqual(float(x[0,100,100]),1);self.assertEqual(float(x[2,100,100]),0)
    def test_d2_normalizacion_distinta(self):
        a=np.zeros((40,80,3),np.uint8);a[:,:,0]=255;c,_,g=letterbox(a,512);x=image_tensor(c)
        self.assertEqual(float(x[0,200,200]),1);self.assertEqual(float(x[2,200,200]),-1);self.assertEqual(g,(6.4,6.4,0,128))
    def test_recorte_margen_y_cuadrado(self):self.assertEqual(crop_box((50,40,150,80),300,200),(38,0,162,124))
    def test_recortes_cercanos_bordes(self):
        for b in [(0,0,5,8),(98,90,100,100),(0,0,99,10),(1,1,3,3)]:
            a=crop_box(b,100,100);self.assertTrue(0<=a[0]<a[2]<=100 and 0<=a[1]<a[3]<=100);self.assertGreaterEqual(min(a[2]-a[0],a[3]-a[1]),16)
    def test_tres_vistas_inversa_espejo_en_imagen_no_padding(self):
        d=NeuralDetector.__new__(NeuralDetector);d.device='cpu';d._lock=threading.Lock();d.model=Mock(return_value=[]);d.nms_iou=.45
        im=np.zeros((100,200,3),np.uint8)
        for size,flip in [(512,False),(640,False),(512,True)]:
            _,_,(sx,sy,l,t)=letterbox(im,size);b=[10,20,50,70] if not flip else [150,20,190,70];b=torch.tensor([[b[0]*sx+l,b[1]*sy+t,b[2]*sx+l,b[3]*sy+t]])
            with patch('ecowaste.vision_equipo.localizador.decode_predictions',return_value=[(b,torch.tensor([.9]))]):r=d._detect_view(im,size,flip,.001)
            np.testing.assert_allclose(r[0]['box'],[10,20,50,70],atol=1e-5)
    def test_fusion_conserva_objetos_proximos_y_suprime_duplicados(self):
        a={'box':[10,10,40,50],'score':.9};b={'box':[42,10,70,50],'score':.8};r=fuse_detection_views([[a,b],[a,b],[a,b]],min_confidence=.27)
        self.assertEqual(len(r),2)
    def test_fusion_sin_objetos(self):self.assertEqual(fuse_detection_views([[],[],[]],min_confidence=.27),[])
    def test_penalizacion_vista_aislada(self):
        self.assertEqual(fuse_detection_views([[{'box':[0,0,20,20],'score':.3}],[],[]],min_confidence=.27),[])
    def fake(self,cuantas):
        m=MotorD2.__new__(MotorD2);m.cfg=leer_config();m.modo='transformer';m.metadatos={'limite':'prueba'};m.detector=Mock();m.banco=Mock()
        m.detector.detect.return_value=[Detection((2.,3.,22.,23.),.7,Region((0,0,32,32),'unit')) for _ in range(cuantas)]
        m.banco.probabilidades.return_value=np.array([[.05,.05,.05,.05,.05,.75]]*cuantas).reshape(cuantas,6)
        return m
    def test_todos_los_objetos_clasificados_no_solo_principal(self):
        m=self.fake(3);r=m.detectar_rgb(np.zeros((64,64,3),np.uint8));self.assertEqual(len(r['detecciones']),3);self.assertEqual(len(m.banco.probabilidades.call_args[0][0]),3)
        self.assertTrue(all(d['material']=='organico' and d['categoria_operativa']=='organico' for d in r['detecciones']))
    def test_no_inventa_cajas_cuando_no_detecta(self):
        m=self.fake(0);r=m.detectar_rgb(np.zeros((64,64,3),np.uint8));self.assertEqual(r['detecciones'],[])
    def test_baja_confianza_no_se_oculta(self):
        m=self.fake(1);m.banco.probabilidades.return_value=np.full((1,6),1/6);d=m.detectar_rgb(np.zeros((64,64,3),np.uint8))['detecciones'][0];self.assertTrue(d['requiere_confirmacion_material']);self.assertIsNone(d['etiqueta_estabilizada'])
    def test_mezcla_65_35_alineada(self):
        b=BancoClasificadores(leer_config());a=np.array([[.8,.1,.1,0,0,0]]);s=np.array([[.1,.3,.2,.2,.1,.1]]);orig=BancoClasificadores.probabilidades
        def f(rec,modo):return a if modo=='refinado' else s if modo=='escena' else orig(b,rec,modo)
        with patch.object(b,'cargar'),patch.object(b,'probabilidades',side_effect=f):r=orig(b,[np.zeros((32,32,3),np.uint8)],'combinado')
        np.testing.assert_allclose(r,.65*a+.35*s)
    def test_rechaza_entrada_no_uint8_rgb(self):
        with self.assertRaises(ValueError):self.fake(0).detectar_rgb(np.zeros((64,64),np.uint8))
    def test_rechaza_caja_invalida(self):
        m=self.fake(1);m.detector.detect.return_value=[Detection((-2,0,10,10),.8,Region((0,0,32,32),'unit'))]
        with self.assertRaises(RuntimeError):m.detectar_rgb(np.zeros((64,64,3),np.uint8))
    def test_ruta_test_protegida(self):
        with self.assertRaises(ValueError):proteger_imagen(Path('datos/test/imagen.jpg'))
    def test_checkpoint_d2_configurado_debe_coincidir_con_manifest(self):
        cfg=leer_config();cfg['detector']=cfg['transformer']
        with self.assertRaisesRegex(ValueError,'otro checkpoint D2'):verificar_pesos(cfg)
    def test_bloqueo_taco_aunque_cambie_mayusculas_extension(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);folder=r/'checkpoints/vision_equipo';folder.mkdir(parents=True);(folder/'reserved_hashes.json').write_text('[]');(folder/'particiones_taco.json').write_text(json.dumps({'por_archivo':{'batch_A/AB.JPG':'test'}}));foto=r/'batch_a/ab.jpg';foto.parent.mkdir();foto.write_bytes(b'archivo de prueba, no fotografia')
            with patch('ecowaste.aplicacion_integrada.RAIZ',r):
                with self.assertRaisesRegex(ValueError,'reservada por D2'):proteger_imagen(foto)
    def test_asistente_no_convierte_material_de_baja_confianza_en_certeza(self):
        m=self.fake(1);m.asistente=Mock();m.asistente.consultar.return_value={'respuesta':'respuesta del modelo','intencion':'material'};r={'detecciones':[{'material':'vidrio','clasificacion_baja_confianza':True}]};a=m.consultar('material',r);self.assertEqual(m.asistente.consultar.call_args[0][1],[]);self.assertEqual(a['objetos_baja_confianza'],1)
    def test_recuperacion_original_no_inventa_dos_confianzas(self):
        base=Mock();base.metadatos={};base.detectar.return_value={'detecciones':[{'bbox_xyxy':[3,3,25,25],'material':'papel','puntuacion':.8}]}
        with patch('ecowaste.video.MotorVideo',return_value=base):r=crear_motor('original').detectar(np.zeros((64,64,3),np.uint8))
        self.assertNotIn('confianza_detector',r['detecciones'][0]);self.assertNotIn('confianza_clasificador',r['detecciones'][0])
        with patch('ecowaste.aplicacion_integrada.cv2.putText') as text:dibujar(np.zeros((128,128,3),np.uint8),r)
        self.assertIn('S:0.80',text.call_args_list[0].args[1])
    def test_video_libera_camara_y_writer_en_q(self):
        cap=Mock();cap.isOpened.return_value=True;cap.read.return_value=(True,np.zeros((64,64,3),np.uint8));cap.get.return_value=10.;wr=Mock();wr.isOpened.return_value=True;m=self.fake(0);m.metadatos={};m.detectar=Mock(return_value={'detecciones':[]})
        with tempfile.TemporaryDirectory() as td,patch('ecowaste.aplicacion_integrada.cv2.VideoCapture',return_value=cap),patch('ecowaste.aplicacion_integrada.cv2.VideoWriter',return_value=wr),patch('ecowaste.aplicacion_integrada.cv2.namedWindow'),patch('ecowaste.aplicacion_integrada.cv2.resizeWindow'),patch('ecowaste.aplicacion_integrada.cv2.imshow'),patch('ecowaste.aplicacion_integrada.cv2.destroyAllWindows'),patch('ecowaste.aplicacion_integrada.cv2.waitKey',return_value=ord('q')):
            r=video(0,m,Path(td)/'salida',camara=True,mostrar=True);self.assertEqual(r['motivo_fin'],'detenido_por_usuario');cap.release.assert_called_once();wr.release.assert_called_once()
    def test_video_libera_camara_si_falla_inferencia(self):
        cap=Mock();cap.isOpened.return_value=True;cap.read.return_value=(True,np.zeros((64,64,3),np.uint8));cap.get.return_value=10.;wr=Mock();wr.isOpened.return_value=True;m=self.fake(0);m.metadatos={};m.detectar=Mock(side_effect=RuntimeError('fallo controlado'))
        with tempfile.TemporaryDirectory() as td,patch('ecowaste.aplicacion_integrada.cv2.VideoCapture',return_value=cap),patch('ecowaste.aplicacion_integrada.cv2.VideoWriter',return_value=wr):
            with self.assertRaisesRegex(RuntimeError,'fallo controlado'):video(0,m,Path(td)/'salida',camara=True)
            cap.release.assert_called_once();wr.release.assert_called_once()
    def test_foto_no_sobrescribe_salida(self):
        with tempfile.TemporaryDirectory() as td:
            with patch('ecowaste.aplicacion_integrada.proteger_imagen'):
                with self.assertRaises(FileExistsError):fotografia(Path('no.jpg'),self.fake(0),Path(td))
if __name__=='__main__':unittest.main()
