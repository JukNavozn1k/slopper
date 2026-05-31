"""Shared faster-whisper model loader.

Loads the Whisper model once (cached per size) and prefers the GPU when one is
available, with a safe fallback to CPU. Used by both the main processing
pipeline (main.py) and the dubbed-video subtitle path (subtitles.py).

Defaults to the most accurate model (large-v3). Override via env vars:
  WHISPER_MODEL        model size/name      (default: large-v3)
  WHISPER_DEVICE       auto | cuda | cpu    (default: auto)
  WHISPER_COMPUTE_TYPE ct2 compute type     (default: int8_float16 on GPU, int8 on CPU)
"""

import os
import sys

_MODEL_CACHE = {}


def _enable_torch_cuda_dlls():
    """On Windows, expose torch's bundled cuDNN/cuBLAS DLLs to CTranslate2.

    The cu128 torch wheel ships the CUDA libraries inside torch/lib, but on
    Windows there is no separate nvidia-cudnn-cu12 package, so CTranslate2
    (faster-whisper's backend) can't find them unless we add that dir to the
    DLL search path.
    """
    if sys.platform != "win32":
        return
    try:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib_dir):
            os.add_dll_directory(lib_dir)
    except Exception:
        pass


def _cuda_available():
    if os.getenv("WHISPER_DEVICE", "auto").lower() == "cpu":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_compute_types():
    """Candidate CUDA compute types, best-first.

    Different GPUs support different paths and CTranslate2 rejects unsupported
    ones at load time, so we probe in order and keep the first that loads:
      - float16: fast + accurate on Turing/Ampere/Ada/Blackwell (RTX 5060)
      - int8: dp4a path that still works on Pascal (GTX 1080, sm_61)
      - float32: universal last resort
    """
    override = os.getenv("WHISPER_COMPUTE_TYPE")
    if override:
        return [override]
    return ["float16", "int8", "float32"]


def get_whisper_model(model_size=None):
    """Return a cached faster-whisper model, on GPU when available else CPU.

    On CUDA the compute type is auto-probed (see _gpu_compute_types) so the same
    code adapts to both Blackwell (RTX 5060) and Pascal (GTX 1080). If no CUDA
    backend can be initialised, falls back to CPU int8.

    NOTE: logs are ASCII-only so they don't crash on the Windows cp1252 console.
    """
    model_size = model_size or os.getenv("WHISPER_MODEL", "large-v3")

    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]

    from faster_whisper import WhisperModel

    forced_device = os.getenv("WHISPER_DEVICE", "auto").lower()
    use_cuda = forced_device == "cuda" or (forced_device == "auto" and _cuda_available())

    model = None
    if use_cuda:
        _enable_torch_cuda_dlls()
        for compute_type in _gpu_compute_types():
            try:
                model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                print(f"[whisper] '{model_size}' loaded on CUDA ({compute_type})")
                break
            except Exception as e:
                print(f"[whisper] CUDA compute_type={compute_type} unavailable: {e}")
        if model is None:
            print("[whisper] no usable CUDA backend; falling back to CPU")

    if model is None:
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        print(f"[whisper] '{model_size}' loaded on CPU ({compute_type})")

    _MODEL_CACHE[model_size] = model
    return model
