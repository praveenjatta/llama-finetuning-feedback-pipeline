import os, time
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_TOKEN"] = "YOUR_HF_KEY"
login(token=os.environ["HF_TOKEN"])

# Model ids and paths
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

fine_tuned_classifier_model_id = "joychak1/Llama-3.1-8B-Instruct-FineTuned-Classifier-v1"
fine_tuned_classifier_adapter_path = "./Llama-3.1-8B-Instruct-FineTuned-Classifier-v1-adapters"
fine_tuned_classifier_model_path = "./Llama-3.1-8B-Instruct-FineTuned-Classifier-v1"

fine_tuned_insightwriter_model_id = "joychak1/Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1"
fine_tuned_insightwriter_adapter_path = "./Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1-adapters"
fine_tuned_insightwriter_model_path = "./Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1"

# This function merges the fine-tuned LoRA adapters back into the base model, saves the merged model locally, and pushes it to Hugging Face Hub.
def merge_to_base_model(fine_tuned_adapter_path, fine_tuned_model_path, fine_tuned_model_id):

    start_time = time.time()
    
    #1. Load the base model and fine-tuned adapters for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu")
    model = PeftModel.from_pretrained(base_model, fine_tuned_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print(f"\n  Base model and fine-tuned adapters loaded successfully for merging!")

    #2. Merge LoRA adapters into the base model
    print(f"\n  Merging LoRA adapters into the base model...")
    model = model.merge_and_unload()
    print(f"\n  LoRA adapters merged into the base model successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    #3. Save the merged model locally
    print(f"\n  Saving the merged model locally...")
    model.save_pretrained(fine_tuned_model_path, max_shard_size="5GB")
    tokenizer.save_pretrained(fine_tuned_model_path)
    print(f"\n  Merged model saved locally at {fine_tuned_model_path} successfully!")
    print(f"----------------------------------------------------------------------------------------------------------------------")

    #4. Push the merged model to Hugging Face Hub
    # print(f"\n  Pushing the merged model to Hugging Face Hub...")
    # model.push_to_hub(fine_tuned_model_id, max_shard_size="5GB")  
    # tokenizer.push_to_hub(fine_tuned_model_id)
    # print("\n  Merged model pushed to Hugging Face Hub successfully!")
    # print(f"----------------------------------------------------------------------------------------------------------------------")

    # calculate and print total time taken for the merging and saving process
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n  Total merging and saving time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

def main():
    print(f"\n  Starting merging process for Classifier model...")
    merge_to_base_model(fine_tuned_classifier_adapter_path, fine_tuned_classifier_model_path, fine_tuned_classifier_model_id)
    print(f"\n  Merging Classifier model process completed successfully!")

    print(f"\n  Starting merging process for Insight Writer model...")
    merge_to_base_model(fine_tuned_insightwriter_adapter_path, fine_tuned_insightwriter_model_path, fine_tuned_insightwriter_model_id)
    print(f"\n  Merging Insight Writer model process completed successfully!")

if __name__ == "__main__":
    main()