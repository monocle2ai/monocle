"""
Support triage agent: jev decides, an OpenAI model replies.

For each customer message:
  1. jev (TypeSafe's decision model) answers: is it about billing? is the tone angry or calm?
  2. gpt-4o-mini drafts a reply using those answers.

The trace has a jev span (entity.2.model_type = decision) and an OpenAI span
(entity.2.model_type = completion) under one agent span.

Setup (reads the repo's .env):
  OPENAI_API_KEY    required
  TYPESAFE_API_KEY  optional; without it a local stand-in answers for jev
  MONOCLE_EXPORTER  optional; e.g. "okahu" or "console" (default: JSON files in ./.monocle)

Run: python examples/jev_support_agent.py
"""
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method, setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.typesafe.methods import DECISION_INFERENCE

load_dotenv()

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this message about billing, charges or refunds?"},
    "tone": {"type": "choice", "instructions": "What is the tone of this message?",
             "criteria": {"angry": "Upset or hostile", "calm": "Neutral or polite"}},
}


class StandInJevClient:
    """Answers like jev from keywords; used until TYPESAFE_API_KEY is set."""
    base_url = "https://api.typesafe.ai"

    def system_one(self, state, questions, *, model=None):
        text = state.lower()
        billing = 0.95 if any(w in text for w in ("charge", "refund", "bill", "invoice")) else 0.05
        angry = 0.85 if any(w in text for w in ("!", "unacceptable", "again", "worst")) else 0.15
        answers = {
            "billing": SimpleNamespace(type="noul", noul=billing),
            "tone": SimpleNamespace(type="choice", choice="angry" if angry > 0.5 else "calm",
                                    confidence=max(angry, 1 - angry),
                                    probabilities={"angry": angry, "calm": 1 - angry}),
        }
        usage = SimpleNamespace(input_tokens=len(text.split()), output_tokens=len(answers))
        return SimpleNamespace(model="jev-standin", usage=usage, answers=answers)


if os.environ.get("TYPESAFE_API_KEY"):
    from typesafe_sdk import TypeSafeClient  # traced by monocle's default TypeSafe instrumentation
    setup_monocle_telemetry(workflow_name="jev_support_agent")
    jev = TypeSafeClient()
else:
    setup_monocle_telemetry(
        workflow_name="jev_support_agent",
        wrapper_methods=[{
            "package": __name__,
            "object": "StandInJevClient",
            "method": "system_one",
            "wrapper_method": task_wrapper,
            "output_processor": DECISION_INFERENCE,
            "span_handler": "non_framework_handler",
        }],
    )
    jev = StandInJevClient()
    print("TYPESAFE_API_KEY not set: using the local jev stand-in.\n")

llm = OpenAI()


@monocle_trace_method(span_name="support_triage_agent")
def handle(message: str) -> str:
    decision = jev.system_one(state=message, questions=QUESTIONS, model="jev-latest")
    is_billing = decision.answers["billing"].noul > 0.5
    tone = decision.answers["tone"].choice
    print(f"jev: billing={is_billing}, tone={tone}")

    route = "the billing team" if is_billing else "general support"
    style = "Apologize first and stay very calm." if tone == "angry" else "Be friendly and brief."
    reply = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a support agent for {route}. {style} Reply in two sentences."},
            {"role": "user", "content": message},
        ],
    )
    return reply.choices[0].message.content


if __name__ == "__main__":
    for message in (
        "You charged my card twice this month again! This is unacceptable.",
        "Hi, how do I change the email on my account?",
    ):
        print(f"customer: {message}")
        print(f"agent: {handle(message)}\n")
    print(f"Traces: exporter = {os.environ.get('MONOCLE_EXPORTER', 'file (./.monocle)')}")
