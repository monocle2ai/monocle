"""
Convenience methods for recording Monocle feedback in traces.

Simple API for capturing user feedback from Kahu agent.
"""

from typing import Any, Dict, Optional
from monocle_apptrace.instrumentation.common.method_wrappers import monocle_trace
from monocle_apptrace.instrumentation.metamodel.feedback.entities.feedback import FEEDBACK


def record_monocle_feedback(
    session_id: str,
    feedback: str,
    turn_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record Monocle feedback as a span in the current trace.
    
    Explicitly invoked API to capture user feedback from Kahu agent.
    
    Args:
        session_id: Session ID (required) - identifies the conversation session
        feedback: Feedback string (required) - the user's feedback text
        turn_id: Turn ID (optional) - identifies specific turn/message in conversation
        
    Returns:
        Dict containing the recorded feedback data
        
    Example:
        >>> # Basic feedback
        >>> record_monocle_feedback(
        ...     session_id="session_123",
        ...     feedback="The response was helpful"
        ... )
        
        >>> # Feedback with turn ID
        >>> record_monocle_feedback(
        ...     session_id="session_123",
        ...     feedback="Great answer!",
        ...     turn_id="turn_456"
        ... )
    """
    # Validate required parameters
    if not session_id:
        raise ValueError("session_id is required")
    if not feedback:
        raise ValueError("feedback is required")
    
    # Build feedback data structure
    feedback_data = {
        "session_id": session_id,
        "feedback": feedback,
        "turn_id": turn_id
    }
    
    # Create span attributes following the FEEDBACK entity structure
    span_attributes = {
        "span.type": "monocle.feedback",
        "entity.1.type": "feedback.monocle",
        "entity.1.session_id": session_id,
    }
    
    if turn_id:
        span_attributes["entity.1.turn_id"] = turn_id
    
    # Create output event with feedback
    events = [
        {
            "name": "data.output",
            "attributes": {
                "feedback": feedback
            }
        }
    ]
    
    # Create the span with monocle_trace context manager
    with monocle_trace(
        span_name="monocle.feedback",
        attributes=span_attributes,
        events=events
    ):
        # The span is automatically created and closed
        pass
    
    return feedback_data


# Legacy function kept for backward compatibility
def record_user_feedback(
    rating: Any,
    comment: Optional[str] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    user_id: Optional[str] = None,
    source: Optional[str] = None,
    rating_type: Optional[str] = None,
    evaluated_response: Optional[str] = None,
    context: Optional[Any] = None,
    sentiment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Legacy function - use record_monocle_feedback for new implementations.
    
    Maps to simplified feedback API.
    """
    # Convert to session_id/turn_id format
    session_id = conversation_id or "unknown_session"
    turn_id = message_id
    
    # Build feedback string from rating and comment
    feedback_parts = []
    if rating:
        feedback_parts.append(f"rating: {rating}")
    if comment:
        feedback_parts.append(f"comment: {comment}")
    
    feedback_str = ", ".join(feedback_parts) if feedback_parts else "no feedback"
    
    return record_monocle_feedback(
        session_id=session_id,
        feedback=feedback_str,
        turn_id=turn_id
    )
