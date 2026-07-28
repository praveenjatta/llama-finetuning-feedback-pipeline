import os, time
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["HF_TOKEN"] = "YOUR_HF_KEY"
login(token=os.environ["HF_TOKEN"])

base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

def download_4bit_model():

    start_time = time.time()
    
    # 1. Load Tokenizer and Model with 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    #2. Load the tokenizer and model from Hugging Face Hub with the specified quantization configuration
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    #3. Load the model with the quantization configuration and automatically map it to available devices
    
    AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    print(f"\n  Model downloaded and loaded with 4-bit quantization successfully!")
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")
    print(f"----------------------------------------------------------------------------------------------------------------------")

def main():
    # This function orchestrates the download and loading of the 4-bit quantized model. It can be extended in the future to include additional setup steps if needed.
    download_4bit_model()
    

if __name__ == "__main__":
    main()