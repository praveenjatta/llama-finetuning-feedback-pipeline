import os
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import time

os.environ["HF_TOKEN"] = "YOUR_HF_KEY"
login(token=os.environ["HF_TOKEN"])

# Model ids and paths
training_classifier_data_path = "./training_data_classifier.jsonl"
training_insightwriter_data_path = "./training_data_insight.jsonl"
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
fine_tuned_classifier_adapter_path = "./Llama-3.1-8B-Instruct-FineTuned-Classifier-v1-adapters"
fine_tuned_InsightWriter_adapter_path = "./Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1-adapters"

# This function handles the entire process of uploading the training data, preparing the model, and fine-tuning it with LoRA adapters for the classifier task.
def upload_and_train_finetune_model(filepath, fine_tuned_adapter_path):

    start_time = time.time()
    
    # 1. Load Tokenizer and Model with 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    print(f"\n  Base model loaded with 4-bit quantization successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    # 2. Load Dataset (example: Alpaca format)
    def format_chat_template(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        return {"text": texts}
    
    print(f"\n  Uploading {filepath}...")
    dataset = load_dataset("json", data_files=filepath, split="train")
    dataset = dataset.map(format_chat_template, batched=True)
    print(f"----------------------------------------------------------------------------------------------------------------------")
    print(f"Total training examples:", len(dataset))
    print(f"First sample data:", dataset[10])
    print(f"----------------------------------------------------------------------------------------------------------------------")

    # 3. Prepare for PEFT (LoRA)
    model = prepare_model_for_kbit_training(base_model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    print(f"\n  LoRA adapters added to the model successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    # 4. Training Arguments
    sft_config = SFTConfig(
        dataset_text_field="messages",
        max_length=2048,
        output_dir = fine_tuned_adapter_path,
        max_steps=100,  # Adjust this based on your dataset size and desired training duration
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    )
    print(f"\n  Training arguments set successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    # 5. Initialize Trainer and Start Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("\n  Starting training...")
    trainer.train()
    print("\n  Training complete and model saved!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    #6. Save the LoRA adapters separately (optional, for later merging)
    print(f"\n  Saving LoRA adapters separately...")
    trainer.save_model(fine_tuned_adapter_path) 
    print(f"\n  LoRA adapters saved to {fine_tuned_adapter_path} successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    #7. Print total training time
    end_time = time.time()
    total_time = end_time - start_time 
    print(f"\n  Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"----------------------------------------------------------------------------------------------------------------------")

def main():
    print(f"\n  Starting fine-tuning process for Classifier model...")
    upload_and_train_finetune_model(training_classifier_data_path, fine_tuned_classifier_adapter_path)
    print("\n  Training Classifier model process completed successfully!")

    print(f"\n  Starting fine-tuning process for Insight Writer model...")
    upload_and_train_finetune_model(training_insightwriter_data_path, fine_tuned_InsightWriter_adapter_path)
    print("\n  Training Insight Writer model process completed successfully!")

if __name__ == "__main__":
    main()