import os, glob, shutil, importlib, pathlib, subprocess, sys

home = os.path.expanduser("~")
jit_root   = os.path.join(home, ".aiter", "jit")
build_root = os.path.join(jit_root, "build")
inst_root  = os.path.join(jit_root, "install")
pkg_root   = os.path.join(inst_root, "private_aiter")
pkg_jit    = os.path.join(pkg_root, "jit")

os.makedirs(pkg_jit, exist_ok=True)
pathlib.Path(os.path.join(pkg_root, "__init__.py")).write_text("")
pathlib.Path(os.path.join(pkg_jit, "__init__.py")).write_text("")

# trigger a build once (ok if it raises)
try:
    import aiter
    from aiter.ops import enum  # will build module_aiter_enum
except Exception as e:
    print("[aiter] prewarm raised:", repr(e))

hits = glob.glob(os.path.join(build_root, "**", "module_aiter_enum*.so"), recursive=True)
if not hits:
    raise SystemExit("[stage] no compiled module_aiter_enum*.so found under " + build_root)

so_src = max(hits, key=os.path.getmtime)
dst = os.path.join(pkg_jit, "module_aiter_enum.so")
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)
try:
    os.symlink(so_src, dst)
    print("[stage] symlinked", dst, "->", so_src)
except OSError:
    shutil.copy2(so_src, dst)
    print("[stage] copied", so_src, "->", dst)

print("[ldd]")
print(subprocess.check_output(["ldd", dst], text=True))

sys.path.insert(0, inst_root)
m = importlib.import_module("private_aiter.jit.module_aiter_enum")
print("[stage] import OK:", m.__spec__.origin)

import aiter; from aiter.ops import enum as _e
print("[stage] aiter import OK")

import asyncio
import multiprocessing as mp

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

# If your model needs HF auth:
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "<your_hf_token>"

import argparse

def build_args(cli_args):
    parser = make_arg_parser(FlexibleArgumentParser())
    return parser.parse_args([
        "--model", cli_args.model,
        "--host", cli_args.host,
        "--port", str(cli_args.port),
        "--gpu-memory-utilization", str(cli_args.gpu_memory_utilization),
        "--max-num-seqs", str(cli_args.max_num_seqs),
        "--max-model-len", str(cli_args.max_model_len),
        # Optional extras:
        # "--served-model-name", "llama-3.2-1b-instruct",
        # "--api-key", "your-secret-key",
    ])

def parse_cli_args():
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    cli_parser.add_argument("--host", type=str, default="0.0.0.0")
    cli_parser.add_argument("--port", type=int, default=18000)
    cli_parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    cli_parser.add_argument("--max-num-seqs", type=int, default=512)
    cli_parser.add_argument("--max-model-len", type=int, default=2048)
    return cli_parser.parse_args()

def main():
    args = build_args(parse_cli_args())
    asyncio.run(run_server(args))

if __name__ == "__main__":
    # On Windows/macOS, Ray/multiprocessing need spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set; safe to ignore
        pass
    # If you're packaging into an EXE, uncomment:
    # mp.freeze_support()
    main()

# test with:
# curl http://nid007966:18080/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"Say hi"}]}'
