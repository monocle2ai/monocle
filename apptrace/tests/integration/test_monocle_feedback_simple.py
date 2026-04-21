"""
Simple example of Monocle feedback instrumentation for Kahu agent.
"""

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.feedback.methods import record_monocle_feedback


def example_simple_feedback():
    """Simple example of recording Monocle feedback."""
    
    # Setup Monocle telemetry
    setup_monocle_telemetry(
        workflow_name="kahu_agent_feedback",
        monocle_exporters_list="console"
    )
    
    # Example 1: Basic feedback with required session_id
    print("Example 1: Basic feedback")
    record_monocle_feedback(
        session_id="session_abc123",
        feedback="The agent response was very helpful and accurate"
    )
    
    # Example 2: Feedback with optional turn_id
    print("\nExample 2: Feedback with turn_id")
    record_monocle_feedback(
        session_id="session_abc123",
        feedback="Great explanation of the concept",
        turn_id="turn_5"
    )
    
    # Example 3: Negative feedback
    print("\nExample 3: Negative feedback")
    record_monocle_feedback(
        session_id="session_xyz789",
        feedback="The response was confusing and didn't answer my question",
        turn_id="turn_2"
    )
    
    # Example 4: Multiple feedbacks in same session
    print("\nExample 4: Multiple feedbacks in same session")
    session_id = "session_multi_001"
    
    record_monocle_feedback(
        session_id=session_id,
        feedback="First turn was good",
        turn_id="turn_1"
    )
    
    record_monocle_feedback(
        session_id=session_id,
        feedback="Second turn was excellent",
        turn_id="turn_2"
    )
    
    record_monocle_feedback(
        session_id=session_id,
        feedback="Third turn could be improved",
        turn_id="turn_3"
    )
    
    print("\n✅ All feedback examples completed!")


def example_in_chatbot_context():
    """Example showing feedback in a chatbot conversation context."""
    
    setup_monocle_telemetry(
        workflow_name="kahu_chatbot",
        monocle_exporters_list="console"
    )
    
    # Simulate a conversation
    session_id = "session_chatbot_001"
    
    print(f"Starting conversation: {session_id}\n")
    
    # Turn 1
    print("User: What is machine learning?")
    print("Agent: Machine learning is a subset of AI...")
    
    # User provides feedback on turn 1
    record_monocle_feedback(
        session_id=session_id,
        feedback="Clear and concise explanation",
        turn_id="turn_1"
    )
    print("✓ Feedback recorded for turn 1\n")
    
    # Turn 2
    print("User: Can you give an example?")
    print("Agent: Sure! Email spam filtering is a common example...")
    
    # User provides feedback on turn 2
    record_monocle_feedback(
        session_id=session_id,
        feedback="Good example, very relatable",
        turn_id="turn_2"
    )
    print("✓ Feedback recorded for turn 2\n")
    
    # Turn 3
    print("User: How does it work technically?")
    print("Agent: At a high level, ML algorithms learn patterns...")
    
    # User provides feedback on turn 3
    record_monocle_feedback(
        session_id=session_id,
        feedback="Too technical, didn't fully understand",
        turn_id="turn_3"
    )
    print("✓ Feedback recorded for turn 3\n")
    
    print(f"✅ Conversation {session_id} completed with feedback!")


def example_validation():
    """Example showing validation of required parameters."""
    
    setup_monocle_telemetry(
        workflow_name="validation_test",
        monocle_exporters_list="console"
    )
    
    # Valid feedback
    try:
        print("Valid feedback:")
        record_monocle_feedback(
            session_id="session_valid",
            feedback="This is valid feedback"
        )
        print("✓ Success\n")
    except ValueError as e:
        print(f"✗ Error: {e}\n")
    
    # Missing session_id
    try:
        print("Missing session_id:")
        record_monocle_feedback(
            session_id="",
            feedback="This should fail"
        )
        print("✓ Success\n")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}\n")
    
    # Missing feedback
    try:
        print("Missing feedback:")
        record_monocle_feedback(
            session_id="session_123",
            feedback=""
        )
        print("✓ Success\n")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Monocle Feedback Examples for Kahu Agent")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Example 1: Simple Feedback Recording")
    print("=" * 80)
    example_simple_feedback()
    
    print("\n" + "=" * 80)
    print("Example 2: Chatbot Conversation Context")
    print("=" * 80)
    example_in_chatbot_context()
    
    print("\n" + "=" * 80)
    print("Example 3: Validation Testing")
    print("=" * 80)
    example_validation()
