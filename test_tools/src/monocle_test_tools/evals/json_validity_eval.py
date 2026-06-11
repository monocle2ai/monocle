import json
import logging
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

class JSONValidityEval(BaseEval):
    """
    Non-LLM evaluator that checks whether the output is well-formed JSON and,
    optionally, whether it conforms to a provided JSON schema.

    eval_options:
        json_schema (dict): Optional JSON schema to validate the parsed output against.
    """
    json_schema: dict = Field(default=None, description="Optional JSON schema to validate against.")

    def __init__(self, **data):
        super().__init__(**data)
        self.json_schema = self.eval_options.get("json_schema", self.json_schema)

    def evaluate(self, eval_args: dict) -> dict:
        return self._json_validity(**eval_args)

    def _json_validity(self, output: str = None, *args, **kwargs) -> dict:
        if output is None:
            raise ValueError("Output must be provided for JSON validity checking.")
        result = {"valid_json": 0.0, "schema_valid": 0.0}
        if isinstance(output, (dict, list)):
            parsed = output
        else:
            try:
                parsed = json.loads(output)
            except (ValueError, TypeError) as e:
                logger.debug("Output is not valid JSON: %s", e)
                return result
        result["valid_json"] = 1.0

        if self.json_schema is None:
            # No schema requested; schema validation is considered satisfied.
            result["schema_valid"] = 1.0
            return result
        try:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=self.json_schema)
            result["schema_valid"] = 1.0
        except jsonschema.ValidationError as e:
            logger.debug("Output does not conform to schema: %s", e.message)
        return result
