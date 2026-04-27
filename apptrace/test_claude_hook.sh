#!/bin/bash
#
# Test script for Claude Code hook package invocation
#
# Usage: ./test_claude_hook.sh
#

set -e

echo "🧪 Testing Claude Code Hook Package Entry Point"
echo ""

# Check if package is installed
if ! python3 -c "import monocle_apptrace" 2>/dev/null; then
    echo "❌ monocle_apptrace not installed"
    echo "   Run: pip install -e ."
    exit 1
fi

echo "✅ monocle_apptrace package found"
echo ""

# Test 1: Check if claude-hook command exists
echo "📋 Test 1: Check if claude-hook command is recognized"
python3 -m monocle_apptrace --help 2>&1 | grep -q "claude-hook" && \
    echo "✅ claude-hook command recognized" || \
    echo "⚠️  claude-hook not in help text (expected)"
echo ""

# Test 2: Create mock event payload
echo "📋 Test 2: Test with mock Stop event"
MOCK_PAYLOAD='{
  "session_id": "test-session-123",
  "transcript_path": "/tmp/test_transcript.jsonl",
  "hook_event_name": "Stop",
  "prompt": "Hello, test!",
  "timestamp": "2024-01-01T00:00:00Z"
}'

# Set required environment variables
export MONOCLE_EXPORTER="console"
export MONOCLE_SERVICE_NAME="claude-cli-test"

# Create dummy transcript file
mkdir -p /tmp
echo '{"type": "user", "message": {"role": "user", "content": "test"}}' > /tmp/test_transcript.jsonl

# Run the hook
echo "$MOCK_PAYLOAD" | python3 -m monocle_apptrace claude-hook

if [ $? -eq 0 ]; then
    echo "✅ Hook executed successfully"
else
    echo "❌ Hook failed"
    exit 1
fi

echo ""
echo "📋 Test 3: Check if session log was created"
if [ -d ".monocle/.claude_sessions" ]; then
    echo "✅ Session directory created: .monocle/.claude_sessions"
    ls -la .monocle/.claude_sessions/ 2>/dev/null || echo "   (no session files yet)"
else
    echo "ℹ️  Session directory not created (may be expected)"
fi

echo ""
echo "📋 Test 4: Check if trace file exists"
if [ -f ".monocle_claude_trace.jsonl" ]; then
    echo "✅ Trace file created"
    echo "   Last 3 lines:"
    tail -3 .monocle_claude_trace.jsonl | head -3
else
    echo "ℹ️  Trace file not created (may be expected)"
fi

echo ""
echo "🎉 All tests completed!"
echo ""
echo "Next steps:"
echo "1. Configure Claude Code:"
echo "   Copy hook command from claude_hook_settings.json"
echo "   to ~/.claude/settings.json or .claude/settings.local.json"
echo ""
echo "2. Create .env file in your project root:"
echo "   export MONOCLE_EXPORTER=\"okahu\""
echo "   export OKAHU_API_KEY=\"your-key\""
echo "   export OKAHU_INGESTION_ENDPOINT=\"https://ingest.okahu.co/api/v1/trace/ingest\""
echo ""
echo "3. Start a Claude Code session and check:"
echo "   - .monocle/.claude_sessions/ for session logs"
echo "   - .monocle_claude_trace.jsonl for all events"
