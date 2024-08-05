from langchain_core.language_models.llms import LLM
from typing import Optional

from typing import Any, List, Mapping


class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""

    responses: List[str]
    """List of responses to return in order."""
    # This parameter should be removed from FakeListLLM since
    # it's only used by sub-classes.
    sleep: Optional[float] = None
    """Sleep time in seconds between responses.

    Ignored by FakeListLLM, but used by sub-classes.
    """
    i: int = 0
    """Internally incremented after every model invocation.

    Useful primarily for testing purposes.
    """
    api_base = ""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}