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
# 1. Install package
pip install monocle_apptrace

# 2. Register hooks into ~/.claude/settings.json
python -m monocle_apptrace claude-install

# 3. Set environment variables in ~/.zshrc (or ~/.bashrc)
export MONOCLE_EXPORTER="okahu,file"
export OKAHU_API_KEY="your-api-key"
export MONOCLE_WORKFLOW_NAME="claude-cli"
```

Start a new Claude Code session — traces flow automatically.

See [Claude Hook Setup Guide](CLAUDE_HOOK_SETUP.md) for complete instructions.
