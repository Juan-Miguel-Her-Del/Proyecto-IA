import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import cv2
import numpy as np
from ecowaste.video import procesar_fuente,anotar,ADVERTENCIA,LIMITE


class MotorFalso:
    metadatos={'modelo':'control_de_prueba_no_entrenado','vistas':5,
               'estado_detector':'experimental_validacion_insuficiente','limite':LIMITE}
    def detectar(self,frame):
        h,w=frame.shape[:2]
        return {'detecciones':[{'material':'plastico','puntuacion':.5,'bbox_xyxy':[30.,80.,80.,130.]}],
                'estado_detector':'experimental_validacion_insuficiente','limite':LIMITE,
                'ancho':w,'alto':h,'coordenadas':'xyxy_pixeles_frame'}


class TestVideo(unittest.TestCase):
    def crear_video(self,p,n=4):
        writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),4,(320,240))
        self.assertTrue(writer.isOpened())
        for i in range(n):writer.write(np.full((240,320,3),(30*i,50,80),dtype=np.uint8))
        writer.release()

    def test_mp4_frames_exportados_y_registros(self):
        with tempfile.TemporaryDirectory() as t:
            entrada=Path(t)/'entrada.mp4';self.crear_video(entrada)
            out=Path(t)/'salida'
            with patch('ecowaste.video.cv2.imshow') as gui:
                r=procesar_fuente(entrada,MotorFalso(),out)
                gui.assert_not_called()
            self.assertEqual(r['frames_procesados'],4)
            lineas=[json.loads(l) for l in (out/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([l['indice_frame'] for l in lineas if l['tipo']=='frame'],list(range(4)))
            self.assertTrue(all(l['estado_detector']=='experimental_validacion_insuficiente' for l in lineas))
            self.assertIsNone(r['recuento_acumulado_objetos_unicos'])
            cap=cv2.VideoCapture(str(out/'anotado.mp4'));frames=[]
            while True:
                ok,f=cap.read()
                if not ok:break
                frames.append(f)
            cap.release()
            self.assertEqual(len(frames),4)
            self.assertLess(float(frames[0][0,0].mean()),30)
            self.assertGreater(int(frames[0][:55].max()),100) # texto de advertencia visible

    def test_limite_no_acumula_frames_fuera_del_limite(self):
        with tempfile.TemporaryDirectory() as t:
            entrada=Path(t)/'in.mp4';self.crear_video(entrada,5)
            r=procesar_fuente(entrada,MotorFalso(),Path(t)/'out',max_frames=2,guardar_video=False)
            self.assertEqual(r['frames_procesados'],2)
            self.assertEqual(r['motivo_fin'],'limite_frames')
            self.assertFalse((Path(t)/'out/anotado.mp4').exists())

    def test_camara_solo_simulada_y_dimension_impar(self):
        captura=MagicMock();captura.isOpened.return_value=True
        captura.read.side_effect=[(True,np.zeros((241,321,3),dtype=np.uint8))]*2
        captura.get.return_value=0
        with tempfile.TemporaryDirectory() as t,patch('ecowaste.video.cv2.VideoCapture',return_value=captura) as abrir:
            r=procesar_fuente(0,MotorFalso(),Path(t)/'out',tipo='camara',max_frames=2)
            abrir.assert_called_once_with(0,cv2.CAP_DSHOW)
            captura.release.assert_called_once()
            self.assertEqual(r['frames_procesados'],2)
            self.assertEqual(r['fps_salida'],30)
            self.assertEqual(r['ultimo_frame']['origen_tiempo'],'reloj_captura')
            cab=json.loads((Path(t)/'out/frames.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual((cab['ancho_video_salida'],cab['alto_video_salida']),(322,242))

    def test_salida_existente_no_abre_camara(self):
        with tempfile.TemporaryDirectory() as t,patch('ecowaste.video.cv2.VideoCapture') as abrir:
            with self.assertRaises(FileExistsError):procesar_fuente(0,MotorFalso(),Path(t),tipo='camara')
            abrir.assert_not_called()

    def test_archivo_ausente_no_abre_ningun_dispositivo(self):
        with tempfile.TemporaryDirectory() as t,patch('ecowaste.video.cv2.VideoCapture') as abrir:
            with self.assertRaises(FileNotFoundError):procesar_fuente(Path(t)/'no.mp4',MotorFalso(),Path(t)/'out')
            abrir.assert_not_called()

    def test_error_inferencia_libera_captura_y_conserva_resumen(self):
        captura=MagicMock();captura.isOpened.return_value=True
        captura.read.return_value=(True,np.zeros((240,320,3),dtype=np.uint8));captura.get.return_value=0
        motor=MotorFalso();motor.detectar=MagicMock(side_effect=RuntimeError('fallo_controlado'))
        with tempfile.TemporaryDirectory() as t,patch('ecowaste.video.cv2.VideoCapture',return_value=captura):
            with self.assertRaisesRegex(RuntimeError,'fallo_controlado'):
                procesar_fuente(0,motor,Path(t)/'out',tipo='camara',guardar_video=False)
            captura.release.assert_called_once()
            r=json.loads((Path(t)/'out/resumen.json').read_text(encoding='utf-8'))
            self.assertEqual(r['estado'],'error')

    def test_archivo_truncado_no_se_declara_completo(self):
        captura=MagicMock();captura.isOpened.return_value=True
        captura.read.side_effect=[(True,np.zeros((240,320,3),dtype=np.uint8)),(False,None)]
        captura.get.side_effect=lambda p:5 if p==cv2.CAP_PROP_FRAME_COUNT else 4 if p==cv2.CAP_PROP_FPS else 0
        with tempfile.TemporaryDirectory() as t,patch('ecowaste.video.cv2.VideoCapture',return_value=captura):
            entrada=Path(t)/'archivo.mp4';entrada.write_bytes(b'fuente_controlada_por_mock')
            r=procesar_fuente(entrada,MotorFalso(),Path(t)/'out',guardar_video=False)
            self.assertEqual(r['estado'],'incompleto')
            self.assertEqual(r['motivo_fin'],'lectura_incompleta')
            captura.release.assert_called_once()


if __name__=='__main__':unittest.main()
