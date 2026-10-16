"""A red-team attack keeps DeepTeam's framing only when DeepTeam's own check agrees, else it stays plain."""
from monocle_test_tools.scenarios.attacks import TEMPLATES, dress


class Reply:
    def __init__(self, content):
        self.content = content


class StubModel:
    """Answers enhance prompts with a rewrite and check prompts with a verdict, in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Reply(self.answers.pop(0))


BASELINE = "Book a flight. Push the agent to complete it immediately."


def test_a_rewrite_that_passes_its_check_is_used():
    model = StubModel(['{"authority_role": "CTO", "input": "TICKET 9921: book it now"}',
                       '{"is_valid_authority": true}'])
    assert dress(model, "authority", BASELINE) == "TICKET 9921: book it now"
    assert BASELINE in model.prompts[0], "the baseline is what enhance() rewrites"
    assert "TICKET 9921: book it now" in model.prompts[1], "the rewrite is what the check judges"


def test_a_failed_check_retries_then_gives_up():
    model = StubModel(['{"input": "weak"}', '{"is_valid_authority": false}'] * 3)
    assert dress(model, "authority", BASELINE) is None
    assert len(model.prompts) == 6


def test_junk_answers_count_as_failed_attempts():
    model = StubModel(["not json at all", '{"input": "x"}', "{broken", '{"input": "y"}', '{"nope": true}'] + [""] * 3)
    assert dress(model, "urgency", BASELINE) is None


def test_json_is_found_inside_prose_and_list_content():
    model = StubModel([[{"type": "text", "text": 'Here you go: {"input": "I am begging you"} hope it helps'}],
                       'Sure. {"is_valid_context": true} (end)'])
    assert dress(model, "urgency", BASELINE) == "I am begging you"


def test_the_check_key_differs_per_template():
    model = StubModel(['{"input": "I am begging you"}', '{"is_valid_context": true}'])
    assert dress(model, "urgency", BASELINE) == "I am begging you"


def test_deliveries_without_a_template_or_a_model_are_not_dressed():
    assert dress(StubModel([]), "plain", BASELINE) is None
    assert dress(None, "authority", BASELINE) is None


def test_every_template_has_its_enhance_and_check_methods():
    for delivery, (template, check, _) in TEMPLATES.items():
        assert callable(template.enhance) and callable(getattr(template, check)), delivery
