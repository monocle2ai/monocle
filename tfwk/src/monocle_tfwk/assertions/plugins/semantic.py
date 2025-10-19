"""
Semantic assertion plugins for semantic similarity validation.

This module contains plugins that provide assertions for validating
semantic similarity of text content using embedding models and
similarity thresholds.
"""
from typing import TYPE_CHECKING

from monocle_tfwk import semantic_similarity
from monocle_tfwk.assertions import trace_utils
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@plugin
class SemanticAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing semantic similarity assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "semantic"
    
    def semantically_contains_input(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Assert that spans contain semantically similar text in input using sentence transformers."""
        matching_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and semantic_similarity(input_text, expected_text, threshold):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with input semantically similar to '{expected_text}' (threshold: {threshold})"
        self._current_spans = matching_spans
        return self
        
    def semantically_contains_output(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Assert that spans contain semantically similar text in output using sentence transformers."""
        matching_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and semantic_similarity(output_text, expected_text, threshold):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with output semantically similar to '{expected_text}' (threshold: {threshold})"
        self._current_spans = matching_spans
        return self
        
    def output_semantically_matches(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Alias for semantically_contains_output for better readability."""
        return self.semantically_contains_output(expected_text, threshold)
    
    def input_semantically_matches(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Alias for semantically_contains_input for better readability."""
        return self.semantically_contains_input(expected_text, threshold)
    
    def semantically_similar_to_any(self, text_options: list, threshold: float = 0.75, 
                                  check_input: bool = True, check_output: bool = True) -> 'TraceAssertions':
        """Assert that spans contain text semantically similar to any of the provided options."""
        matching_spans = []
        
        for span in self._current_spans:
            found_match = False
            
            if check_input:
                input_text = trace_utils.get_input_from_span(span)
                if input_text:
                    for option in text_options:
                        if semantic_similarity(input_text, option, threshold):
                            found_match = True
                            break
            
            if not found_match and check_output:
                output_text = trace_utils.get_output_from_span(span)
                if output_text:
                    for option in text_options:
                        if semantic_similarity(output_text, option, threshold):
                            found_match = True
                            break
            
            if found_match:
                matching_spans.append(span)
        
        assert matching_spans, (
            f"No spans found semantically similar to any of {text_options} (threshold: {threshold})"
        )
        self._current_spans = matching_spans
        return self
    
    def assert_semantic_coherence(self, threshold: float = 0.5) -> 'TraceAssertions':
        """Assert that inputs and outputs within spans are semantically coherent."""
        incoherent_spans = []
        
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            output_text = trace_utils.get_output_from_span(span)
            
            if input_text and output_text:
                similarity = semantic_similarity(input_text, output_text, 0.0)  # Get raw score
                if similarity < threshold:
                    incoherent_spans.append((span.name, similarity))
        
        assert not incoherent_spans, (
            f"Found spans with low input/output semantic coherence (threshold: {threshold}): "
            f"{[(name, f'{score:.3f}') for name, score in incoherent_spans]}"
        )
        return self
    
    def assert_output_semantic_diversity(self, min_threshold: float = 0.3, max_threshold: float = 0.8) -> 'TraceAssertions':
        """Assert that outputs have appropriate semantic diversity (not too similar, not too different)."""
        if len(self._current_spans) < 2:
            return self  # Need at least 2 spans to compare
        
        outputs = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text:
                outputs.append((span.name, output_text))
        
        if len(outputs) < 2:
            return self
        
        similarity_violations = []
        
        # Compare all pairs of outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                name1, text1 = outputs[i]
                name2, text2 = outputs[j]
                
                similarity = semantic_similarity(text1, text2, 0.0)  # Get raw score
                
                if similarity < min_threshold:
                    similarity_violations.append((f"{name1}-{name2}", similarity, "too_different"))
                elif similarity > max_threshold:
                    similarity_violations.append((f"{name1}-{name2}", similarity, "too_similar"))
        
        assert not similarity_violations, (
            f"Output semantic diversity violations (range: [{min_threshold}, {max_threshold}]): "
            f"{[(pair, f'{score:.3f}', issue) for pair, score, issue in similarity_violations]}"
        )
        return self