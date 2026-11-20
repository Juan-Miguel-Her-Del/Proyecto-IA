"""Diagnóstico real de Python/PyTorch/CUDA. No descarga pesos ni entrena modelos."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def diagnosticar():
    informe = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "ejecutable": sys.executable,
        "sistema": platform.platform(),
        "tipo_prueba": "operaciones_sinteticas_sin_modelo_entrenado",
        "correcto": False,
    }
    try:
        proceso = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=20, check=False
        )
        informe["nvidia_smi"] = proceso.stdout.strip() or proceso.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        informe["nvidia_smi"] = str(error)
    informe["paquetes"] = {}
    for paquete in ["torch", "numpy", "pillow", "opencv-python", "pyyaml"]:
        try:
            informe["paquetes"][paquete] = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError:
            informe["paquetes"][paquete] = None
    informe["importaciones"] = {}
    for modulo in ["numpy", "PIL", "cv2", "yaml"]:
        try:
            importlib.import_module(modulo)
            informe["importaciones"][modulo] = "correcta"
        except Exception as error:
            informe["importaciones"][modulo] = f"{type(error).__name__}: {error}"
    try:
        import torch
        import torch.nn.functional as F

        informe["torch"] = torch.__version__
        informe["cuda_runtime_torch"] = torch.version.cuda
        informe["cuda_disponible"] = torch.cuda.is_available()
        if not informe["cuda_disponible"]:
            raise RuntimeError("PyTorch no puede utilizar CUDA.")
        torch.manual_seed(42)
        propiedades = torch.cuda.get_device_properties(0)
        informe["gpu"] = propiedades.name
        informe["vram_total_bytes"] = propiedades.total_memory
        informe["capacidad_cuda"] = list(torch.cuda.get_device_capability(0))
        torch.cuda.reset_peak_memory_stats()

        # Comparación independiente CPU/GPU: comprueba valores, no solo ejecución.
        matriz_cpu = torch.randn(128, 128)
        esperado = matriz_cpu @ matriz_cpu
        matriz_gpu = matriz_cpu.to("cuda")
        obtenido = matriz_gpu @ matriz_gpu
        torch.cuda.synchronize()
        torch.testing.assert_close(obtenido.cpu(), esperado, rtol=1e-3, atol=1e-3)
        informe["multiplicacion_gpu_vs_cpu"] = "correcta"

        # Tamaño ilustrativo del futuro asistente: B=2, H=4, T=64, D_cabeza=16.
        tensores = [torch.randn(2, 4, 64, 16, device="cuda", requires_grad=True) for _ in range(3)]
        salida = F.scaled_dot_product_attention(*tensores, dropout_p=0.0)
        perdida = salida.square().mean()
        perdida.backward()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(salida).all()):
            raise RuntimeError("La atención produjo valores no finitos.")
        for tensor in tensores:
            if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
                raise RuntimeError("Se encontraron gradientes ausentes o no finitos.")
            if tensor.grad.abs().sum().item() == 0:
                raise RuntimeError("Se encontraron gradientes nulos.")
        informe["atencion_forward_backward"] = "correcta"
        informe["forma_atencion"] = list(salida.shape)
        informe["pico_memoria_asignada_prueba_bytes"] = torch.cuda.max_memory_allocated()
        informe["nota_memoria"] = "Solo mide esta prueba sintética; no estima el entrenamiento completo."
        if any(estado != "correcta" for estado in informe["importaciones"].values()):
            raise RuntimeError("Hay dependencias de imagen/configuración ausentes o que no se pueden importar.")
        informe["correcto"] = True
    except Exception as error:
        informe["error"] = f"{type(error).__name__}: {error}"
    return informe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    informe = diagnosticar()
    contenido = json.dumps(informe, ensure_ascii=False, indent=2)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(contenido + "\n", encoding="utf-8")
    print(contenido)
    return 0 if informe["correcto"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
