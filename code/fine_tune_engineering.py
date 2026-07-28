"""
Fine-Tuning: Feedback-to-Engineering-Insights (2 Models)
=========================================================
Run with: python fine_tune_engineering.py

Prerequisites:
  pip install openai
  Set OPENAI_API_KEY as environment variable (do NOT hardcode)
    Windows PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."
    Mac/Linux:           export OPENAI_API_KEY="sk-proj-..."
"""
import sys, os, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Set OPENAI_API_KEY environment variable first."); sys.exit(1)

client = OpenAI()
BASE_MODEL = "gpt-4o-mini-2024-07-18"
N_EPOCHS = 3

def validate_jsonl(filepath):
    print(f"\n  Validating {filepath}...")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            roles = [m["role"] for m in data["messages"]]
            assert roles[0] == "system", f"Line {i}: first must be system"
            assert "user" in roles, f"Line {i}: needs user"
            assert "assistant" in roles, f"Line {i}: needs assistant"
            count += 1
    print(f"  PASSED - {count} examples")
    return count

def upload_and_train(filepath, suffix):
    print(f"\n  Uploading {filepath}...")
    with open(filepath, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"  File ID: {file_obj.id}")
    print(f"  Creating job (suffix: {suffix})...")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id, model=BASE_MODEL,
        hyperparameters={"n_epochs": N_EPOCHS}, suffix=suffix)
    print(f"  Job ID: {job.id}")
    return job.id

def wait_for_job(job_id, name):
    print(f"\n  Waiting for {name}...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status == "succeeded":
            print(f"  {name} COMPLETE: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ("failed", "cancelled"):
            print(f"  {name} {job.status.upper()}: {getattr(job, 'error', 'unknown')}")
            return None
        else:
            trained = ""
            if hasattr(job, "trained_tokens") and job.trained_tokens:
                trained = f" | Tokens: {job.trained_tokens}"
            print(f"  [{name}] {job.status}{trained}")
            time.sleep(30)

def test_model(model_id, name, system_prompt, test_input):
    print(f"\n  Testing {name}...")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": test_input}])
    output = response.choices[0].message.content
    print(f"  Input: '{test_input[:80]}...'")
    print(f"  Output (first 400 chars):")
    print(f"  {'-'*50}")
    for line in output[:400].split('\n'):
        print(f"  {line}")
    print(f"  {'-'*50}")

def main():
    print("=" * 60)
    print("FINE-TUNING: Feedback to Engineering Insights (2 Models)")
    print("=" * 60)

    print("\n--- MODEL 1: Feedback Classifier ---")
    validate_jsonl("training_data_classifier.jsonl")
    classifier_job = upload_and_train("training_data_classifier.jsonl", "fb-classifier")

    print("\n--- MODEL 2: Engineering Insight Writer ---")
    validate_jsonl("training_data_insight.jsonl")
    insight_job = upload_and_train("training_data_insight.jsonl", "eng-insight")

    print("\n" + "=" * 60)
    print("WAITING FOR BOTH JOBS (running in parallel on OpenAI)")
    print("=" * 60)

    classifier_model = wait_for_job(classifier_job, "Classifier")
    insight_model = wait_for_job(insight_job, "Insight Writer")

    if classifier_model:
        test_model(classifier_model, "Classifier",
            "You are a user feedback classifier for engineering teams. If feedback is vague, set CATEGORY to UNCLEAR.",
            "This app is garbage. Total waste of money.")
    if insight_model:
        test_model(insight_model, "Insight Writer",
            "You are an engineering insight writer. If CATEGORY is UNCLEAR, write 'Insufficient feedback for engineering action'.",
            "Original feedback: This app is garbage.\n\nClassification:\nCATEGORY: UNCLEAR\nSEVERITY: LOW\nAFFECTED_SYSTEM: UNKNOWN")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if classifier_model:
        print(f"\n  Classifier model:  {classifier_model}")
        print(f"  -> Paste into 'Classifier LLM' node in n8n")
    if insight_model:
        print(f"\n  Insight Writer model:  {insight_model}")
        print(f"  -> Paste into 'Insight LLM' node in n8n")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open n8n workflow")
    print("2. Click 'Classifier LLM' node -> change model to classifier ID above")
    print("3. Click 'Insight LLM' node -> change model to insight writer ID above")
    print("4. Save workflow")
    print("5. Run all 8 test feedback items again")
    print("6. Paste new outputs into test_feedback_engineering.py")
    print("7. Run: python test_feedback_engineering.py")
    print("8. Compare before vs after")

if __name__ == "__main__":
    main()