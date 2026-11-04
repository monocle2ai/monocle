"""Runs a custom eval template inside Monocle, as five explicit pipeline steps, one verdict per trace."""
from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan

from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.monocle import judge, preprocess, results, template as template_loader
from monocle_test_tools.evals.monocle.judge import RunResult
from monocle_test_tools.evals.monocle.preprocess import Evidence
from monocle_test_tools.evals.monocle.template import EvalTemplate, LlmConfig

OPTION_KEYS = ("trace_source", "repeat", "min_agreement", "model")
DEFAULT_REPEAT = 1
DEFAULT_MIN_AGREEMENT = 0.8
TRACES = "traces"
BARE_NAME_MESSAGE = "'monocle' runs custom templates only; pass a template file path"

Traces = dict[str, list[ReadableSpan]]


class MonocleEval(BaseEval):
    last_judge_output: dict = {}
    last_total_tokens: int = 0
    last_fact_results: list = []
    last_judge_description: str = ""

    def __init__(self, **data):
        """Raise ValueError when eval_options holds a key other than trace_source, repeat, min_agreement or model,
        min_agreement is not a number above 0 and at most 1, or model is not a local model spec."""
        super().__init__(**data)
        self.eval_options = self.eval_options or {}
        _check_option_keys(self.eval_options)
        _check_min_agreement(self.eval_options.get("min_agreement", DEFAULT_MIN_AGREEMENT))
        _check_model(self.eval_options.get("model"))

    def evaluate(
        self,
        filtered_spans: Optional[list[ReadableSpan]] = [],
        eval_name: Optional[str] = "",
        fact_name: Optional[str] = TRACES,
        eval_args: dict = {},
        template: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Orchestrate the eval as five explicit steps:
            1. Validate the template
            2. Load the raw data: the traces in the spans
            3. Preprocess each trace into judge-ready evidence
            4. Run the judge over each trace's evidence
            5. Process the runs into verdicts, and the verdicts into what check_eval reads back

        Returns the last trace's (label, explanation). Raises ValueError for a bare template name or
        a fact_name other than "traces", and AssertionError when the spans hold no trace or a
        trace has no evidence.
        """
        eval_template = self._validate_template(template)
        spans_by_trace = self._load_eval_data(filtered_spans, fact_name)
        evidence_by_trace = self._preprocess(spans_by_trace, eval_template)
        runs_by_trace = self._run_llm_evaluation(evidence_by_trace, eval_template)
        return self._process_results(runs_by_trace, eval_template, filtered_spans)

    def cleanup(self):
        """Nothing to clean up: the eval never leaves this process."""

    def _validate_template(self, template: Optional[dict]) -> EvalTemplate:
        if not template:
            raise ValueError(BARE_NAME_MESSAGE)
        return template_loader.load(template)

    def _load_eval_data(self, spans: list[ReadableSpan], fact_name: Optional[str]) -> Traces:
        if fact_name and fact_name != TRACES:
            raise ValueError(f'with_evaluation("monocle") judges traces; fact_name="{fact_name}" is not supported')
        spans_by_trace: Traces = {}
        for span in spans:
            spans_by_trace.setdefault(_trace_id_of(span), []).append(span)
        if not spans_by_trace:
            raise AssertionError("No spans to judge.")
        return spans_by_trace

    def _preprocess(self, spans_by_trace: Traces, eval_template: EvalTemplate) -> dict[str, Evidence]:
        evidence_by_trace = {}
        for trace_id, trace_spans in spans_by_trace.items():
            evidence = preprocess.build_evidence(trace_spans, trace_id, eval_template.evidence_scope)
            if evidence is None:
                raise AssertionError(
                    f"No evidence for trace {trace_id}: no span is left inside the evidence_scope of '{eval_template.name}'."
                )
            evidence_by_trace[trace_id] = evidence
        return evidence_by_trace

    def _run_llm_evaluation(self, evidence_by_trace: dict[str, Evidence], eval_template: EvalTemplate) -> dict[str, list[RunResult]]:
        trace_judge = judge.judge_for(self._judge_llm(eval_template.llm))
        return {
            trace_id: judge.run(trace_judge, eval_template, evidence, repeat=self._repeat())
            for trace_id, evidence in evidence_by_trace.items()
        }

    def _process_results(self, runs_by_trace: dict[str, list[RunResult]], eval_template: EvalTemplate, spans: list[ReadableSpan]) -> tuple[str, str]:
        verdicts = [
            results.settle(runs, eval_template, trace_id, min_agreement=self._min_agreement())
            for trace_id, runs in runs_by_trace.items()
        ]
        workflow_name = spans[0].attributes.get("workflow.name") or ""
        self.last_judge_description = self._judge_description(eval_template)
        self.last_fact_results = [results.row_for_check_eval(verdict, workflow_name) for verdict in verdicts]
        self.last_judge_output = verdicts[-1].judge_output
        self.last_total_tokens = sum(verdict.total_tokens for verdict in verdicts)
        return verdicts[-1].label, verdicts[-1].explanation

    def _judge_description(self, eval_template: EvalTemplate) -> str:
        llm = self._judge_llm(eval_template.llm)
        if judge.is_decision_model(llm):
            return f"{llm.model}, min_confidence {llm.min_confidence:.2f}"
        runs = self._repeat()
        return f"{llm.model}, {runs} run{'s' if runs != 1 else ''} per trace"

    def _judge_llm(self, llm: LlmConfig) -> LlmConfig:
        """The template's llm, or the local model named by the model option in its place."""
        local_model = self.eval_options.get("model")
        if not local_model:
            return llm
        if judge.is_decision_model(llm):
            raise ValueError(f'model="{local_model}" cannot run a decision-model template; "{llm.model}" has no local form')
        return llm.model_copy(update={"model": local_model})

    def _repeat(self) -> int:
        return self.eval_options.get("repeat", DEFAULT_REPEAT)

    def _min_agreement(self) -> float:
        return self.eval_options.get("min_agreement", DEFAULT_MIN_AGREEMENT)


def _trace_id_of(span: ReadableSpan) -> str:
    return format(span.get_span_context().trace_id, "032x")


def _check_option_keys(eval_options: dict) -> None:
    unknown_keys = sorted(set(eval_options) - set(OPTION_KEYS))
    if unknown_keys:
        raise ValueError(f"Unknown eval_options for 'monocle': {unknown_keys}. Accepted: {list(OPTION_KEYS)}")


def _check_model(model) -> None:
    from monocle_test_tools.harness.local_model import is_local_spec

    if model is None:
        return
    if not is_local_spec(model):
        raise ValueError(f'model= takes "local" or "local:<name>", got {model!r}; a hosted judge is named by the template\'s llm.model')


def _check_min_agreement(min_agreement) -> None:
    if isinstance(min_agreement, bool) or not isinstance(min_agreement, (int, float)):
        raise ValueError(f"min_agreement must be a number above 0 and at most 1, got {min_agreement!r}")
    if not 0 < min_agreement <= 1:
        raise ValueError(f"min_agreement must be a number above 0 and at most 1, got {min_agreement!r}")
