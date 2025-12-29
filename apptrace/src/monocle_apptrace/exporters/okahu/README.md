# Okahu Integration - List Applications Feature

This feature allows you to list and query applications registered on the Okahu observability platform directly from the command line or Python code.

## Installation

The feature is included in the `monocle_apptrace` package:

```bash
pip install monocle_apptrace
```

## Configuration

Set your Okahu API key as an environment variable:

```bash
export OKAHU_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

#### List all applications

```bash
# Table format (default)
monocle-okahu list-apps

# JSON format
monocle-okahu list-apps --format json
```

**Example Output (Table Format):**
```
Found 4 application(s):

App Name                       App ID                                   Status         
-------------------------------------------------------------------------------------
AI Travel Agent                app-001                                  active         
Customer Service Bot           app-002                                  active         
Document Analyzer              app-003                                  active         
Code Review Assistant          app-004                                  inactive       
```

#### Get details for a specific application

```bash
monocle-okahu get-app app-001
```

**Example Output:**
```json
{
  "id": "app-001",
  "name": "AI Travel Agent",
  "status": "active",
  "workflow_name": "travel-agent-workflow",
  "created_at": "2025-01-01T00:00:00Z",
  "traces_count": 1500,
  "last_trace_at": "2025-01-15T12:00:00Z"
}
```

### Python API

```python
from monocle_apptrace.exporters.okahu import OkahuClient

# Initialize the client (uses OKAHU_API_KEY from environment)
with OkahuClient() as client:
    # List all applications
    apps = client.list_apps()
    for app in apps:
        print(f"App: {app['name']} (ID: {app['id']})")
    
    # Get details for a specific app
    app_details = client.get_app("app-001")
    print(f"Details: {app_details}")
```

### Alternative: Using Custom API Key

```python
from monocle_apptrace.exporters.okahu import OkahuClient

# Pass API key directly
with OkahuClient(api_key="your_api_key") as client:
    apps = client.list_apps()
```

## Features

- **List Applications**: View all registered applications on Okahu platform
- **Get Application Details**: Retrieve detailed information for specific apps
- **Multiple Output Formats**: Support for table and JSON formats
- **Error Handling**: Graceful error handling with clear messages
- **Resource Management**: Automatic cleanup using context managers
- **Environment Variable Support**: Configure using standard environment variables

## Environment Variables

- `OKAHU_API_KEY` (required): Your Okahu API key
- `OKAHU_API_BASE_URL` (optional): Custom API base URL (defaults to https://api.okahu.co)

## Examples

See the [examples](./examples/) directory for complete working examples:

- `list_okahu_apps.py` - Complete script demonstrating list apps functionality

## Testing

Run the test suite:

```bash
cd apptrace
python -m pytest tests/unit/test_okahu*.py -v
```

All tests should pass:
- Unit tests for OkahuClient
- Integration tests for end-to-end functionality
- Error handling and edge case tests

## Troubleshooting

### "OKAHU_API_KEY not set" error

Make sure you've set the environment variable:
```bash
export OKAHU_API_KEY=your_api_key_here
```

### Connection errors

Check that you can reach the Okahu API:
```bash
curl -H "x-api-key: your_api_key" https://api.okahu.co/api/v1/apps
```

## Related Documentation

- [Okahu Platform Documentation](https://docs.okahu.ai/)
- [Monocle User Guide](../../Monocle_User_Guide.md)
- [Okahu Exporter](./okahu_exporter.py)
