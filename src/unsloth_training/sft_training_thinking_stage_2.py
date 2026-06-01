from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import re
import multiprocessing as mp
from trl import SFTTrainer, SFTConfig

max_seq_length = 8192 # Can increase for longer reasoning traces
# lora_rank = 32 # Larger rank = smarter, but slower, 8, 16, 32, 64, 128, depends on your GPU VRAM
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

# Load model
model, processor = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-35B-A3B", # This is a very big model, might take a while for downloading # You can use any model from the list above and HF will download it for you. Depends on your GPU memory.
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False, # Not supported for MoE (yet!)
)
tokenizer = processor.tokenizer # To tokenize tex

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
)

dataset = load_dataset("ChaosAIVision/Ensure-Thamdo-1000-training", split = "train")

# Build format temlate with messages columns
def _strip(x):
    return (x or "").strip()


def normalize_assistant_to_think_solution(text: str) -> str:
    text = _strip(text)

    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end():]
        rest = rest.lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
    else:
        return f"<think></think>\n{text}".rstrip()

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        for convo in convos
    ]
    return {"text": texts}


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2, # Number of samples processed on each device (GPU) in one forward/backward pass.
        gradient_accumulation_steps = 4,  # Accumulate gradients over 6 steps before updating weights; simulates a larger batch size without needing more VRAM.
        warmup_ratio = 0.04,  # Use the first 3%-5% of total training steps to gradually increase the learning rate for more stable training.
        #warmup_steps = 60,
        num_train_epochs = 3, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_steps = 100,  # Save a checkpoint every 100 training steps.You can adjust this as needed.
        save_total_limit = 3, # Keep only the most recent checkpoint; older ones are deleted to save disk space.
        save_strategy = "steps",
        # report_to = "wandb", # Can use Weights & Biases
        output_dir = '/home/chaos/Documents/chaos/project/spec_training/spec_training/checkpoints/sft/qwen3-thamdo-thinking',
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n<think>",
)

# Compilation can take 2-3 minutes of time, so please be patient :)
trainer.train()
# If training is interrupted, you can resume with:
# trainer.train(resume_from_checkpoint=True)  # auto-load latest checkpoint or trainer.train(resume_from_checkpoint="checkpoint-xxx")  # specify a checkpoint path