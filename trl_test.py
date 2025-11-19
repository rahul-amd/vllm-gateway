import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


# Get vllm server host from environment variable
vllm_server_host = os.getenv("VLLM_SERVER_HOST", "nid007974")
vllm_server_port = os.getenv("VLLM_SERVER_PORT", "8080")

training_args = GRPOConfig(
    output_dir="trl_test_output",
    use_vllm=True,
    vllm_server_host=vllm_server_host,
    vllm_server_port=vllm_server_port,
    per_device_train_batch_size=2,
    max_completion_length=1024, # Only for testing purposes 
    num_generations=2,
    max_steps=10,  # Only for testing purposes
    )

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

# Dummy reward function for demonstration purposes
def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


trainer = GRPOTrainer(
    model="meta-llama/Llama-3.2-1B-Instruct",
    reward_funcs=reward_num_unique_letters,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
