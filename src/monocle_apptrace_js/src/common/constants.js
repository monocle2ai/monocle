exports.WORKFLOW_TYPE_KEY = "workflow_type"
exports.CONTEXT_INPUT_KEY = "workflow_context_input"
exports.CONTEXT_OUTPUT_KEY = "workflow_context_output"
exports.PROMPT_INPUT_KEY = "workflow_input"
exports.PROMPT_OUTPUT_KEY = "workflow_output"
exports.CONTEXT_PROPERTIES_KEY = "workflow_context_properties"
exports.LLM_INPUT_KEY = "llm_input"
exports.LLM_OUT_PUT_KEY = "llm_output"

exports.WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack"
}