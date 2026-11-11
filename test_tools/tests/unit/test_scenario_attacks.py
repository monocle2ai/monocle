"""A red-team attack takes DeepTeam's rewrite only when DeepTeam's own check passes it; otherwise it stays plain."""
from monocle_test_tools.scenarios.attacks import TEMPLATES, dress

BASELINE = "Book a flight. Push the agent to complete it immediately."


class Model:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Reply", (), {"content": self.answers.pop(0)})()


def test_a_rewrite_that_passes_its_check_is_used():
    model = Model(['{"input": "TICKET 9921: book it now"}', 'Sure. {"is_valid_authority": true}'])
    assert dress(model, "authority", BASELINE) == "TICKET 9921: book it now"
    assert BASELINE in model.prompts[0] and "TICKET 9921" in model.prompts[1]


def test_three_failed_checks_or_junk_give_no_rewrite():
    assert dress(Model(['{"input": "weak"}', '{"is_valid_authority": false}'] * 3), "authority", BASELINE) is None
    assert dress(Model(["not json", "{broken", ""]), "urgency", BASELINE) is None


def test_plain_or_no_model_means_no_rewrite():
    assert dress(Model([]), "plain", BASELINE) is None and dress(None, "authority", BASELINE) is None


def test_every_template_has_its_enhance_and_check():
    for delivery, (template, check, _) in TEMPLATES.items():
        assert callable(template.enhance) and callable(getattr(template, check)), delivery
