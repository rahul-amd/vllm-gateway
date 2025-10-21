### Start trl vllm-serve

Get a node for trl vllm-serve.
```
srun \
    --account=project_462000963 \
    --partition=dev-g \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --gres=gpu:mi250:1 \
    --time=0-02:00:00 \
    --mem=64G \
    --hint=nomultithread \
    --pty bash
```

Drop into singularity container as follows:
```
CONTAINER=/scratch/project_462000353/danizaut/containers/vllm_v11.0_pytorch_v2.8.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd,/usr
alias sing='singularity shell -B "$PWD" '"$CONTAINER"
sing
```

Set/unset the following environment variables:
```
export PATH="/opt/miniconda3/envs/pytorch/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH"
export PYTHON=/opt/miniconda3/envs/pytorch/bin/python
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_ROCM_USE_AITER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
export HF_HOME=/pfs/lustrep2/scratch/project_462000353/aralikatte/.cache/huggingface
unset TRANSFORMERS_CACHE
unset ROCR_VISIBLE_DEVICES
mkdir -p $TORCH_EXTENSIONS_DIR
HIP_VISIBLE_DEVICES=0   # <- set this to the GPU you want to use
```
Next, install trl: `pip install -U trl`. It should install in your `~/.local`. Modify your `~/.local/bin/trl` as such:
```
#!/opt/miniconda3/envs/pytorch/bin/python3.10
import sys
import torch
from trl.cli import main
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
```

Then start trl vllm-serve: `trl vllm-serve --model Qwen/Qwen2.5-7B`.

### Start vllm-client

Get another node in the same way as above, and drop into singularity container (and set the same environment variables).

Then run the following code:
````
from trl.extras.vllm_client import VLLMClient

HOST="10.253.1.200"
PORT="8000"
client = VLLMClient(base_url=f"http://{HOST}:{PORT}")
client.generate(["Hello, AI!", "Tell me a joke"])

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
client.init_communicator(device="cuda")
client.update_model_params(model)
```

You can also start both the server and client in the same node, but make sure that they use different GPUs.
