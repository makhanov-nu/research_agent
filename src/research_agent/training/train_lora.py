"""Reference recipe: LoRA fine-tune a small base on one role's exported dataset.

This is a *reference*, not part of the runtime — it needs the heavy `train` extra
(`pip install -e ".[train]"`) and a GPU, so its deps are imported lazily and it's
never imported by the package. Run it on your compute box, once per role:

    python -m research_agent.training.train_lora \
        --data outputs/datasets/methodology.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --out adapters/methodology

Each role gets its own LoRA adapter over a *shared* base model; serve them with
multi-LoRA (e.g. vLLM) and route by agent kind. That's the controllable, cheap
"experts" alternative to a MoE we settled on: one base in memory + tiny adapters,
and you decide which adapter handles which subagent.
"""

from __future__ import annotations

import argparse


def main() -> None:
    # Heavy deps imported lazily so importing the package never requires them.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser(description="LoRA fine-tune one role's dataset.")
    parser.add_argument("--data", required=True, help="JSONL from research-agent-export")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="base model")
    parser.add_argument("--out", required=True, help="output dir for the LoRA adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The exporter writes {"messages": [...]} per line; TRL applies the base
    # model's chat template to these automatically.
    dataset = load_dataset("json", data_files=args.data, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved LoRA adapter for this role to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
