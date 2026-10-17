"""Each goal sentence is split into its details by one model call, with names safe to use as tool names."""
from monocle_test_tools.scenarios.seed import GoalSplit, MinimalSeedFile, ParamSplit, expand


class StubModel:
    def __init__(self, splits):
        self.splits, self.prompts = list(splits), []

    def with_structured_output(self, schema):
        assert schema is GoalSplit
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.splits.pop(0)


def test_each_goal_is_split_into_its_details():
    model = StubModel([GoalSplit(
        id="book_flight", goal="Book a flight from one airport to another", success="Booked.", side_effect=True,
        untrusted_input=True,
        params=[ParamSplit(name="source", value="Bombay", description="source airport"),
                ParamSplit(name="destination", value="Hyderabad", description="destination airport")])])
    seeds = expand(MinimalSeedFile(agent="A travel agent.", goals=["Book a flight from Bombay to Hyderabad"],
                                   out_of_scope=["refunds"]), model)
    assert "Book a flight from Bombay to Hyderabad" in model.prompts[0]
    seed = seeds.seeds[0]
    assert seed.id == "book_flight" and seed.side_effect and seed.untrusted_input
    assert seed.params["destination"].value == "Hyderabad"
    assert seeds.out_of_scope == ["refunds"]


def test_names_from_the_model_are_made_identifier_safe_and_unique():
    split = lambda: GoalSplit(id="Book a Hotel!", goal="g", success="s", side_effect=False, untrusted_input=False, params=[
        ParamSplit(name="check-in date", value="a", description="d"),
        ParamSplit(name="check in date", value="b", description="d"),
        ParamSplit(name="2nd guest", value="c", description="d")])
    seeds = expand(MinimalSeedFile(agent="a", goals=["x", "y"]), StubModel([split(), split()]))
    assert [s.id for s in seeds.seeds] == ["book_a_hotel", "book_a_hotel_2"]
    assert list(seeds.seeds[0].params) == ["check_in_date", "check_in_date_2", "detail_2nd_guest"]
