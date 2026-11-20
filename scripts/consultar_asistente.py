"""Consulta el asistente local usando detecciones suministradas en un JSON."""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.asistente import Asistente

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pregunta")
parser.add_argument("--detecciones", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, default=RAIZ / "checkpoints" / "asistente_v1.pt")
args = parser.parse_args()
entrada = json.loads(args.detecciones.read_text(encoding="utf-8"))
resultado = Asistente(args.checkpoint).consultar(args.pregunta, entrada["detecciones"])
resultado["origen_detecciones"] = entrada.get("origen", "suministradas_por_usuario")
if entrada.get('estado_detector'):
    resultado['estado_detector']=entrada['estado_detector']
    resultado['limite_detector']=entrada.get('limite')
print(json.dumps(resultado, ensure_ascii=False, indent=2))
