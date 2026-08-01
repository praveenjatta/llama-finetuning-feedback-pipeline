"""
run_full_eval.py  —  Option B automation (direct local model calls, no n8n, no ngrok)

Replicates the two-agent n8n pipeline by calling the local FastAPI server directly:
  Feedback Classifier (/v2)  ->  Engineering Insight Writer (/v3)

Runs all 8 test cases (F1-F8), collects the final insight-writer output for each,
and writes them to outputs.properties under the correct phase keys.

USAGE:
  1. Start the matching model server first:
       - BEFORE fine-tuning:  python3.11 load_run_local_model.py        (base, port 8005, /v1)
       - AFTER  fine-tuning:  python3.11 load_run_local_finetune_model.py (tuned, port 8006, /v2 + /v3)
  2. Then run this with the matching phase:
       python3 run_full_eval.py --phase before
       python3 run_full_eval.py --phase after

Outputs are written to outputs.properties as Before_fine_tuning_F1..F8 / After_fine_tuning_F1..F8.
After both phases are written, run the evaluation:  python3 test_feedback_engineering.py --phase before|after
"""
import sys, argparse, time
import requests

from props_loader import write_outputs, FKEYS
from agent_prompts import CLASSIFIER_SYSTEM, INSIGHT_SYSTEM

# ------------------------------------------------------------
# The 8 test inputs (same as test_feedback_engineering.py TEST_FEEDBACK)
# ------------------------------------------------------------
TEST_INPUTS = {
    "F1": "The checkout page keeps freezing when I try to apply a discount code on my iPhone 14. Happens every time. I've lost 3 orders this week because of it. Using Safari, latest iOS. -- App Store review, 1 star",
    "F2": "The API response times have degraded significantly since last month. Our p95 latency went from 200ms to 800ms. We're on the Enterprise plan and this is affecting our production systems. We've opened 3 tickets about this already. Account: api-team@bigcorp.com -- Support ticket",
    "F3": "The onboarding flow is confusing. I signed up yesterday and still can't figure out how to create my first project. The getting started guide links to a 404 page. I watched 3 YouTube tutorials and they all show a different UI than what I see. Free plan, Chrome on Windows. -- Support ticket",
    "F4": "Login alert at 3:47 AM EST from IP 203.0.113.42 (Romania). I did NOT log in. Account: lisa.t@company.com, #AC-41209. Changed password, enabled 2FA. Need: 1) Was data accessed? 2) 30-day audit log 3) Other accounts affected? -- Support ticket",
    "F5": "This app is garbage. Total waste of money.",
    "F6": "Terrible. Just terrible. 0 stars if I could.",
    "F7": "Meh. It's fine I guess.",
    "F8": "Fix your stuff.",
}

# ------------------------------------------------------------
# Server config per phase
#   before = base model server (single model, /v1)
#   after  = fine-tuned server (two models, /v2 classifier + /v3 writer)
# ------------------------------------------------------------
PHASE_CONFIG = {
    "before": {
        "base_url": "http://127.0.0.1:8005",
        "classifier_endpoint": "/v1/chat/completions",
        "insight_endpoint":    "/v1/chat/completions",  # base server has one model for both roles
    },
    "after": {
        "base_url": "http://127.0.0.1:8006",
        "classifier_endpoint": "/v2/chat/completions",  # fine-tuned classifier
        "insight_endpoint":    "/v3/chat/completions",  # fine-tuned insight writer
    },
}

# generation params (matches what we settled on in n8n to avoid timeouts)
TEMPERATURE = 0.3
MAX_TOKENS = 800
REQUEST_TIMEOUT = 300  # seconds — local 8B model can be slow


def call_agent(url, system_prompt, user_content):
    """Send one chat completion request to the local server and return the text."""
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def run_pipeline(fk, feedback, cfg):
    """Run one feedback item through classifier -> insight writer. Returns final insight text."""
    classifier_url = cfg["base_url"] + cfg["classifier_endpoint"]
    insight_url = cfg["base_url"] + cfg["insight_endpoint"]

    # Agent 1: classify
    classification = call_agent(classifier_url, CLASSIFIER_SYSTEM, feedback)

    # Agent 2: write insight from (original feedback + classification)
    writer_input = f"ORIGINAL FEEDBACK:\n{feedback}\n\nCLASSIFIER OUTPUT:\n{classification}"
    insight = call_agent(insight_url, INSIGHT_SYSTEM, writer_input)

    return insight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["before", "after"],
                        help="which model is running: 'before' (base) or 'after' (fine-tuned)")
    parser.add_argument("--test-one", action="store_true",
                        help="run ONLY F1 to validate the model call works, then stop (does not write the file)")
    args = parser.parse_args()
    cfg = PHASE_CONFIG[args.phase]

    print("=" * 60)
    print(f"RUN FULL EVAL — phase: {args.phase.upper()}")
    print(f"Server: {cfg['base_url']}  classifier={cfg['classifier_endpoint']}  insight={cfg['insight_endpoint']}")
    print("=" * 60)

    # --- test-one mode: run just F1, print the result, do NOT write the file ---
    if args.test_one:
        fk = "F1"
        feedback = TEST_INPUTS[fk]
        print(f"\n[TEST-ONE] running {fk} only...  \"{feedback[:55]}...\"")
        t0 = time.time()
        try:
            result = run_pipeline(fk, feedback, cfg)
        except Exception as e:
            print(f"\n[TEST-ONE] ERROR: {e}")
            print("The model call failed — check the server is running and the endpoints match this phase.")
            sys.exit(1)
        dt = time.time() - t0
        print(f"\n[TEST-ONE] SUCCESS in {dt:.1f}s. Model returned {len(result)} chars.")
        print("-" * 60)
        print(result[:800])
        print("-" * 60)
        print("\nIf that output looks like a proper engineering insight, the pipeline works.")
        print(f"Now run the full 8:  python3 run_full_eval.py --phase {args.phase}")
        return

    outputs = {}
    for fk in FKEYS:
        feedback = TEST_INPUTS[fk]
        print(f"\n[{fk}] running...  \"{feedback[:55]}...\"")
        t0 = time.time()
        try:
            outputs[fk] = run_pipeline(fk, feedback, cfg)
            dt = time.time() - t0
            print(f"[{fk}] done in {dt:.1f}s  ({len(outputs[fk])} chars)")
        except Exception as e:
            print(f"[{fk}] ERROR: {e}")
            print("Aborting — make sure the correct server is running for this phase.")
            sys.exit(1)

    # write all 8 to the properties file under the phase keys
    write_outputs(args.phase, outputs, path="outputs.properties")
    print("\n" + "=" * 60)
    print(f"WROTE 8 outputs to outputs.properties as {args.phase.capitalize()}_fine_tuning_F1..F8")
    print("Next: python3 test_feedback_engineering.py --phase " + args.phase)
    print("=" * 60)


if __name__ == "__main__":
    main()
