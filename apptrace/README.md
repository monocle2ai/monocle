# Monocle Apptrace

**Monocle** helps developers and platform engineers building or managing GenAI apps monitor these in prod by making it easy to instrument their code to capture traces that are compliant with open-source cloud-native observability ecosystem. 

**Monocle** is a community-driven OSS framework for tracing GenAI app code governed as a [Linux Foundation AI & Data project](https://lfaidata.foundation/projects/monocle/). 

## Use Monocle

- Get the Monocle package
  
```
    pip install monocle_apptrace 
```
- Instrument your app code
     - Import the Monocle package
       ```
          from monocle_apptrace.instrumentor import setup_monocle_telemetry
       ```
     - Setup instrumentation in your ```main()``` function  
       ``` 
          setup_monocle_telemetry(workflow_name="your-app-name")
       ```         
- (Optionally) Modify config to alter where traces are sent

See [Monocle user guide](Monocle_User_Guide.md) for more details.

## Claude Code Instrumentation

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install package
uv tool install monocle_apptrace

# 3. Register hooks — prompts for Okahu API key (leave blank for local-only)
monocle-apptrace claude-setup
```

Start a new Claude Code session — traces flow automatically.

See [Hook Setup Guide](HOOK_SETUP.md) for complete instructions.

## Codex CLI Instrumentation

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install package
uv tool install monocle_apptrace

# 3. Register hooks — prompts for Okahu API key (leave blank for local-only)
monocle-apptrace codex-setup
```

Start a new Codex session — traces flow automatically.

See [Hook Setup Guide](HOOK_SETUP.md) for complete instructions.
