#!/usr/bin/env python3
"""
Quick test for enhanced flow pattern parser with parentheses support.
"""
import pytest
from monocle_tfwk.assertions.flow_validator import FlowPattern


def test_enhanced_parser():
    """Test that enhanced parser can handle parentheses."""
    
    print("Testing Enhanced Flow Pattern Parser")
    print("=" * 50)
    
    # Test cases that should now work
    test_patterns = [
        "A -> B",  # Simple sequence
        "A || B",  # Simple parallel
        "A -> (B || C)",  # Sequence with parallel group
        "validate_user_data -> (calculate_user_profile_score -> send_notification)",  # From our test
        "http.process -> validate_* -> (calculate_* -> send_*) -> audit_*",  # Complex pattern
        "(A || B) -> C",  # Parallel then sequence
        "A -> (B -> C) -> D",  # Nested sequences
        "A? -> B*"  # Optional and wildcard
    ]
    
    for i, pattern in enumerate(test_patterns, 1):
        print(f"\n{i}. Testing pattern: '{pattern}'")
        try:
            flow_pattern = FlowPattern(
                name=f"test_{i}",
                pattern=pattern,
                description=f"Test pattern {i}"
            )
            
            parsed = flow_pattern.parsed_pattern
            print("   ✅ Successfully parsed!")
            print(f"   Steps: {parsed.get('steps', [])}")
            print(f"   Operators: {parsed.get('operators', [])}")
            
            # Check if AST is available
            if 'ast' in parsed:
                print(f"   AST available: {parsed['ast']['type']}")
            
        except Exception as e:
            print(f"   ❌ Failed to parse: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])