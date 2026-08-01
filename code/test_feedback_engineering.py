"""
test_feedback_engineering.py  —  DeepEval evaluation, reads outputs from outputs.properties

CLEANED UP: no more hardcoded/commented PIPELINE_OUTPUTS blocks.
Outputs come from outputs.properties via the --phase flag.

USAGE:
  python3 test_feedback_engineering.py --phase before
  python3 test_feedback_engineering.py --phase after

Prerequisites:
  pip install deepeval openai requests
  Set OPENAI_API_KEY as an environment variable (do NOT hardcode):
    export OPENAI_API_KEY="sk-proj-..."
"""
import sys, os, argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from props_loader import get_outputs

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable first.")
    print('  export OPENAI_API_KEY="sk-proj-..."')
    sys.exit(1)

# ------------------------------------------------------------
# Test inputs + label (detailed vs vague). No outputs here anymore —
# outputs are loaded from outputs.properties by phase.
# ------------------------------------------------------------
TEST_FEEDBACK = {
    "F1": {"input": "The checkout page keeps freezing when I try to apply a discount code on my iPhone 14. Happens every time. I've lost 3 orders this week because of it. Using Safari, latest iOS. -- App Store review, 1 star", "label": "detailed"},
    "F2": {"input": "The API response times have degraded significantly since last month. Our p95 latency went from 200ms to 800ms. We're on the Enterprise plan and this is affecting our production systems. We've opened 3 tickets about this already. Account: api-team@bigcorp.com -- Support ticket", "label": "detailed"},
    "F3": {"input": "The onboarding flow is confusing. I signed up yesterday and still can't figure out how to create my first project. The getting started guide links to a 404 page. I watched 3 YouTube tutorials and they all show a different UI than what I see. Free plan, Chrome on Windows. -- Support ticket", "label": "detailed"},
    "F4": {"input": "Login alert at 3:47 AM EST from IP 203.0.113.42 (Romania). I did NOT log in. Account: lisa.t@company.com, #AC-41209. Changed password, enabled 2FA. Need: 1) Was data accessed? 2) 30-day audit log 3) Other accounts affected? -- Support ticket", "label": "detailed"},
    "F5": {"input": "This app is garbage. Total waste of money.", "label": "vague"},
    "F6": {"input": "Terrible. Just terrible. 0 stars if I could.", "label": "vague"},
    "F7": {"input": "Meh. It's fine I guess.", "label": "vague"},
    "F8": {"input": "Fix your stuff.", "label": "vague"},
}

# ------------------------------------------------------------
# METRICS (unchanged)
# ------------------------------------------------------------
engineering_groundedness = GEval(
    name="Engineering Groundedness",
    criteria=(
        "The input is detailed user feedback with specific technical details. "
        "Does the engineering insight ONLY use information from this feedback? "
        "1) REPRODUCTION STEPS must only describe actions the user actually mentioned. "
        "2) No invented internal system names or architecture details the user did not mention "
        "(e.g., do not invent 'API Gateway' or 'GraphQL' if the user just said 'API'). "
        "3) No fabricated user counts, percentages, or revenue impact. "
        "4) DATA GAPS section should list what is NOT known. "
        "Grounded = PASS. Invented technical specifics = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

engineering_usefulness = GEval(
    name="Engineering Usefulness",
    criteria=(
        "Would an engineer find this insight useful for starting an investigation? "
        "Does it identify the affected system based on what the user described? "
        "Does it provide a reasonable investigation checklist? "
        "Does it clearly state data gaps? "
        "Useful engineering ticket = PASS. Vague or unhelpful = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

vague_handling = GEval(
    name="Vague Feedback Handling",
    criteria=(
        "The input is vague or lacks technical details. A CORRECT response should: "
        "1) Identify that engineering cannot act without more information. "
        "2) Set affected components to UNKNOWN. "
        "3) NOT invent systems, platforms, error types, or root causes. "
        "4) NOT fabricate reproduction steps. "
        "5) Recommend reaching out to the user for clarification. "
        "6) Set priority to LOW since nothing is actionable. "
        "The checklist CAN include multiple follow-up items about GATHERING info "
        "(ask for platform, request screenshots) as long as no specific problem is invented. "
        "Correctly flagging as unactionable = PASS. Inventing details = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")


def run_evaluation(phase):
    # load the 8 outputs for this phase from the properties file
    outputs = get_outputs(phase, path="outputs.properties")

    # identifier makes runs show up cleanly (and comparable) in the DeepEval dashboard
    run_id = f"llama-{phase}-finetuning"

    print("\n" + "=" * 60)
    print(f"EVALUATING DETAILED FEEDBACK (4 cases) — phase: {phase}")
    print("=" * 60)
    detailed = [LLMTestCase(input=d["input"], actual_output=outputs[f])
                for f, d in TEST_FEEDBACK.items() if d["label"] == "detailed"]
    if detailed:
        evaluate(test_cases=detailed,
                 metrics=[engineering_groundedness, engineering_usefulness, relevancy],
                 identifier=run_id)

    print("\n" + "=" * 60)
    print(f"EVALUATING VAGUE FEEDBACK (4 cases) — phase: {phase}")
    print("=" * 60)
    vague = [LLMTestCase(input=d["input"], actual_output=outputs[f])
             for f, d in TEST_FEEDBACK.items() if d["label"] == "vague"]
    if vague:
        evaluate(test_cases=vague,
                 metrics=[vague_handling, relevancy],
                 identifier=run_id)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE — phase:", phase)
    print("=" * 60)
    print("\nWhat to look for:")
    print("  Detailed: Groundedness + Usefulness should PASS")
    print("  Vague:    Vague Handling should PASS")
    print("  All:      Relevancy should PASS")
    print(f"\nDashboard run identifier: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["before", "after"],
                        help="which phase's outputs to evaluate from outputs.properties")
    args = parser.parse_args()

    print("=" * 60)
    print("FEEDBACK TO ENGINEERING INSIGHTS — DeepEval Evaluation")
    print("=" * 60)
    run_evaluation(args.phase)
