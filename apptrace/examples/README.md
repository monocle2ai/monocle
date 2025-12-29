# Monocle Examples

This directory contains example scripts demonstrating various features of Monocle.

## Okahu Integration Examples

### Listing Applications

The `list_okahu_apps.py` script demonstrates how to use the Okahu client to list applications registered on the Okahu platform.

**Prerequisites:**
- Okahu API key (set as `OKAHU_API_KEY` environment variable)

**Usage:**
```bash
export OKAHU_API_KEY=your_api_key_here
python examples/list_okahu_apps.py
```

**Features:**
- Lists all applications registered on Okahu
- Displays app name, ID, and status
- Shows how to get detailed information for a specific app

### Using the CLI

You can also use the Monocle CLI tool to list apps:

```bash
# List all applications
monocle-okahu list-apps

# List in JSON format
monocle-okahu list-apps --format json

# Get details for a specific app
monocle-okahu get-app <app-id>
```
