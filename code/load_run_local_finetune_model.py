import time
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI(title="Local Llama 3.1 Unbreakable Server")

MODEL_ID_CLASSIFIER = "joychak1/Llama-3.1-8B-Instruct-FineTuned-Classifier-v1"
fine_tuned_model_path_classifier = "./Llama-3.1-8B-Instruct-FineTuned-Classifier-v1"

MODEL_ID_INSIGHTWRITER = "joychak1/Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1"
fine_tuned_model_path_insightwriter = "./Llama-3.1-8B-Instruct-FineTuned-InsightWriter-v1"

def load_model(model_path: str, model_id: str):
    print(f"Loading {model_id} model from local cache: {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return model, tokenizer, llm_pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

model_classifier, tokenizer_classifier, llm_pipeline_classifier = load_model(fine_tuned_model_path_classifier, MODEL_ID_CLASSIFIER)
model_insightwriter, tokenizer_insightwriter, llm_pipeline_insightwriter = load_model(fine_tuned_model_path_insightwriter, MODEL_ID_INSIGHTWRITER)

def clean_message_list(raw_list: list) -> List[Dict[str, str]]:
    """Transforms any variation of a list safely into clean Llama 3.1 message dictionaries."""
    cleaned = []
    for item in raw_list:
        # Case 1: Item is already a dictionary {"role": "...", "content": "..."}
        if isinstance(item, dict):
            role = str(item.get("role", "user")).lower()
            content = str(item.get("content", item.get("text", "")))
            
            if role in ["human", "user", "system"]: role = "user"
            if role in ["ai", "assistant"]: role = "assistant"
            
            cleaned.append({"role": role, "content": content})
            
        # Case 2: Item is a nested list pair like ["human", "hello"]
        elif isinstance(item, list):
            if len(item) >= 2:
                # CRITICAL FIX: Safe integer indexing on the inner item
                role = str(item[0]).lower()
                content = str(item[1])
                
                if role in ["human", "user", "system"]: role = "user"
                if role in ["ai", "assistant"]: role = "assistant"
                
                cleaned.append({"role": role, "content": content})
            elif len(item) == 1:
                cleaned.append({"role": "user", "content": str(item[0])})
                
        # Case 3: Item is a flat string inside the list ["Hello there"]
        elif isinstance(item, str):
            role = "assistant" if len(cleaned) % 2 != 0 else "user"
            cleaned.append({"role": role, "content": item})
            
    return cleaned


def run_inference(model_id, tokenizer, llm_pipeline, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Generates text using the local Llama 3.1 pipeline."""
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Run the Hugging Face pipeline
    outputs = llm_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True if temperature > 0 else False,
        temperature=temperature if temperature > 0 else None,
    )
    
    # CRITICAL FIX: The pipeline returns a LIST of dicts: [{"generated_text": "..."}]
    # Calling outputs["generated_text"] causes the exact 'list indices must be integers' error.
    if isinstance(outputs, list) and len(outputs) > 0:
        raw_output = outputs[0]
    else:
        raw_output = outputs
        
    if isinstance(raw_output, dict) and "generated_text" in raw_output:
        full_text = raw_output["generated_text"]
    else:
        full_text = str(raw_output)

    # Strip the prompt text away from the generated answer safely
    generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text

    print(f"----------------------------------------------------------------------------------------------------")
    print(f"Generated text before cleanup: {generated_text}")
    print(f"----------------------------------------------------------------------------------------------------")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(generated_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(generated_text))
        }
    }

async def handle_any_payload(request: Request, model_id, tokenizer, llm_pipeline):
    try:
        body = await request.json()
        print("\n================= INCOMING RAW PAYLOAD =================")
        import json
        print(json.dumps(body, indent=2))
        print("========================================================\n")
    except Exception:
        raise HTTPException(status_code=400, detail="Payload is not valid JSON.")

    messages = []
    temperature = 0.3
    max_tokens = 800

    try:
        # 1. Parse Parameters safely
        if isinstance(body, dict):
            try:
                raw_temp = body.get("temperature", 0.6)
                temperature = float(raw_temp) if raw_temp is not None else 0.6
            except: temperature = 0.6
            try:
                raw_tokens = body.get("max_tokens", body.get("max_new_tokens", max_tokens))
                max_tokens = int(raw_tokens) if raw_tokens is not None else max_tokens
            except: max_tokens = max_tokens

        # 2. Extract underlying text query (Deep scanning)
        extracted_prompt = None

        if isinstance(body, dict):
            # Check standard array keys first
            if "messages" in body and body["messages"]:
                messages = clean_message_list(body["messages"])
            elif "prompt" in body and body["prompt"]:
                if isinstance(body["prompt"], list):
                    messages = clean_message_list(body["prompt"])
                else:
                    extracted_prompt = str(body["prompt"])
            elif "text" in body and body["text"]:
                extracted_prompt = str(body["text"])
            
            # If still nothing, extract the first non-empty string or value found
            if not messages and not extracted_prompt:
                for key, val in body.items():
                    if key not in ["temperature", "max_tokens", "max_new_tokens", "stream", "model"]:
                        if val and str(val).strip() and str(val) != "{}":
                            extracted_prompt = str(val)
                            break
        elif isinstance(body, list):
            messages = clean_message_list(body)
        else:
            extracted_prompt = str(body)

        # Build fallback user message if array wasn't populated directly
        if not messages and extracted_prompt:
            # Clean string if it accidentally contains stringified empty brackets
            clean_str = extracted_prompt.strip()
            if clean_str == "{}" or clean_str == "":
                clean_str = "Hello! Please give me a prompt."
            messages = [{"role": "user", "content": clean_str}]

        # Final absolute fallback to prevent the model from getting empty inputs
        if not messages or (len(messages) == 1 and messages[0]["content"].strip() in ["{}", ""]):
            messages = [{"role": "user", "content": "Hello Llama! Give me a quick greeting test status."}]

        if temperature < 0.0: temperature = 0.0
        if max_tokens <= 0: max_tokens = 10000

        print(f"----------------------------------------------------------------------------------------------------")
        print(f"Final extracted messages for model input: {messages}")
        print(f"----------------------------------------------------------------------------------------------------")

        return run_inference(model_id, tokenizer, llm_pipeline, messages, temperature, max_tokens)

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(
            status_code=400,
            content={"error": "Deep Parse Failure", "detail": str(e), "traceback": tb_str}
        )

@app.post("/v2/chat/completions")
async def handle_any_payload_classifier(request: Request):
    return await handle_any_payload(request, MODEL_ID_CLASSIFIER, tokenizer_classifier, llm_pipeline_classifier)

@app.post("/v3/chat/completions")
async def handle_any_payload_insightwriter(request: Request):
    return await handle_any_payload(request, MODEL_ID_INSIGHTWRITER, tokenizer_insightwriter, llm_pipeline_insightwriter)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006, timeout_keep_alive=1800, reload=False)
