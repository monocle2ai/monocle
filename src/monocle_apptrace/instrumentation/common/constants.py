# Azure environment constants
AZURE_ML_ENDPOINT_ENV_NAME = "AZUREML_ENTRY_SCRIPT"
AZURE_FUNCTION_WORKER_ENV_NAME = "FUNCTIONS_WORKER_RUNTIME"
AZURE_APP_SERVICE_ENV_NAME = "WEBSITE_SITE_NAME"
AWS_LAMBDA_ENV_NAME = "AWS_LAMBDA_RUNTIME_API"
GITHUB_CODESPACE_ENV_NAME = "CODESPACES"

AWS_LAMBDA_FUNCTION_IDENTIFIER_ENV_NAME = "AWS_LAMBDA_FUNCTION_NAME"
AZURE_FUNCTION_IDENTIFIER_ENV_NAME = "WEBSITE_SITE_NAME"
AZURE_APP_SERVICE_IDENTIFIER_ENV_NAME = "WEBSITE_DEPLOYMENT_ID"
GITHUB_CODESPACE_IDENTIFIER_ENV_NAME = "GITHUB_REPOSITORY"


# Azure naming reference can be found here
# https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/resource-abbreviations
# https://docs.aws.amazon.com/resource-explorer/latest/userguide/supported-resource-types.html#services-lookoutmetrics
AZURE_FUNCTION_NAME = "azure_func"
AZURE_APP_SERVICE_NAME = "azure_webapp"
AZURE_ML_SERVICE_NAME = "azure_ml"
AWS_LAMBDA_SERVICE_NAME = "aws_lambda"
GITHUB_CODESPACE_SERVICE_NAME = "github_codespace"

# Env variables to identify infra service type
service_type_map = {
    AZURE_ML_ENDPOINT_ENV_NAME: AZURE_ML_SERVICE_NAME,
    AZURE_APP_SERVICE_ENV_NAME: AZURE_APP_SERVICE_NAME,
    AZURE_FUNCTION_WORKER_ENV_NAME: AZURE_FUNCTION_NAME,
    AWS_LAMBDA_ENV_NAME: AWS_LAMBDA_SERVICE_NAME,
    GITHUB_CODESPACE_ENV_NAME: GITHUB_CODESPACE_SERVICE_NAME
}

# Env variables to identify infra service name
service_name_map = {
    AZURE_APP_SERVICE_NAME: AZURE_APP_SERVICE_IDENTIFIER_ENV_NAME,
    AZURE_FUNCTION_NAME: AZURE_FUNCTION_IDENTIFIER_ENV_NAME,
    AZURE_ML_SERVICE_NAME: AZURE_ML_ENDPOINT_ENV_NAME,
    AWS_LAMBDA_SERVICE_NAME: AWS_LAMBDA_FUNCTION_IDENTIFIER_ENV_NAME,
    GITHUB_CODESPACE_SERVICE_NAME: GITHUB_CODESPACE_IDENTIFIER_ENV_NAME
}


llm_type_map = {
    "sagemakerendpoint": "aws_sagemaker",
    "azureopenai": "azure_openai",
    "openai": "openai",
    "chatopenai": "openai",
    "azurechatopenai": "azure_openai",
    "bedrock": "aws_bedrock",
    "sagemakerllm": "aws_sagemaker",
    "chatbedrock": "aws_bedrock",
    "openaigenerator": "openai",
    "bedrockruntime":"aws_bedrock",
    "sagemakerruntime":"aws_sagemaker",
    "anthropic": "anthropic",
    "chatanthropic":"anthropic",
    "anthropicchatgenerator":"anthropic",
    "chatcompletionsclient": "azure_ai_inference",
    "embeddingsclient": "azure_ai_inference",
    "imageembeddingsclient": "azure_ai_inference",
    "chatbedrockconverse": "aws_bedrock",
    "googleaigeminichatgenerator": "gemini",
    "gemini": "gemini",
    "chatgooglegenerativeai": "gemini",
    "azurechatcompletion": "azure_openai",
    "openaichatcompletion": "openai",
}

MONOCLE_INSTRUMENTOR = "monocle_apptrace"
WORKFLOW_TYPE_KEY = "workflow_type"
DATA_INPUT_KEY = "data.input"
DATA_OUTPUT_KEY = "data.output"
PROMPT_INPUT_KEY = "data.input"
PROMPT_OUTPUT_KEY = "data.output"
QUERY = "input"
RESPONSE = "response"
SESSION_PROPERTIES_KEY = "session"
INFRA_SERVICE_KEY = "infra_service_name"
META_DATA = 'metadata'
MONOCLE_SCOPE_NAME_PREFIX = "monocle.scope."
SCOPE_METHOD_LIST = 'MONOCLE_SCOPE_METHODS'
SCOPE_METHOD_FILE = 'monocle_scopes.json'
SCOPE_CONFIG_PATH = 'MONOCLE_SCOPE_CONFIG_PATH'
TRACE_PROPOGATION_URLS = "MONOCLE_TRACE_PROPAGATATION_URLS"
WORKFLOW_TYPE_KEY = "monocle.workflow_type"
ADD_NEW_WORKFLOW = "monocle.add_new_workflow"
WORKFLOW_TYPE_GENERIC = "workflow.generic"
MONOCLE_SDK_VERSION = "monocle_apptrace.version"
MONOCLE_SDK_LANGUAGE = "monocle_apptrace.language"
MONOCLE_DETECTED_SPAN_ERROR = "monocle_apptrace.detected_span_error"
HTTP_SUCCESS_CODES = ('200', '201', '202', '204', '205', '206')
CHILD_ERROR_CODE = "child.error.code"

AGENT_PREFIX_KEY = "monocle.agent.prefix"

# agentic sub types
INFERENCE_AGENT_DELEGATION = "delegation"
INFERENCE_TOOL_CALL = "tool_call"
INFERENCE_COMMUNICATION = "turn"

SCOPE_NAME = "scope_name"
AGENT_INVOCATION_SPAN_NAME = "agentic.invocation"
AGENT_REQUEST_SPAN_NAME = "agentic.request"

AGENTIC_SPANS = [AGENT_INVOCATION_SPAN_NAME, AGENT_REQUEST_SPAN_NAME]

# Span sub types

## OPTIONAL right next to span.type,  span.subtype:
## subtype is one perspective , are non overlapping, limitations: only one classification scheme for subtypes
#1 planning
SPAN_SUBTYPE_PLANNING = "planning"

#2 routing and selection INFERENCE_TOOL_CALL, INFERENCE_AGENT_DELEGATION
SPAN_SUBTYPE_ROUTING = "routing"

#3 content processing
SPAN_SUBTYPE_CONTENT_PROCESSING = "content_processing"

#4 content generation
SPAN_SUBTYPE_CONTENT_GENERATION = "content_generation"

#5 communication INFERENCE_COMMUNICATION
SPAN_SUBTYPE_COMMUNICATION = "communication"

#6 transformations , if structured output
SPAN_SUBTYPE_TRANSFORMATIONS = "transformations"

#7 domain specific,
SPAN_SUBTYPE_DOMAIN_SPECIFIC = "domain_specific"

#8 generic (we may skip this property)
SPAN_SUBTYPE_GENERIC = "generic"

# ALL span.type:
# generic

# agentic.delegation
# agentic.tool.invocation
# agentic.invocation
# agentic.mcp.invocation
# agentic.request(TBD**)

# RAG pattern => text => embedding => search
# embedding (explicit embedding)
# retrieval (there is implicit embedding, may not be instrumented separately)

# workflow
# inference.*( "", framework, modelapi) 

# http.send
# http.process

## OPTIONAL right next to span.type,  span.subtype:
## subtype is one perspective , are non overlapping, limitations: only one classification scheme for subtypes
#1 planning
#2 routing and selection INFERENCE_TOOL_CALL, INFERENCE_AGENT_DELEGATION
#3 content processing
#4 content generation
#5 communication INFERENCE_COMMUNICATION
#6 transformations , if structured output
#7 domain specific, 
#8 generic (we may skip this property)

# agent as tools in openai agents sdk etc.


##
# {
#                 "name": "metadata",
#                 "timestamp": "2025-07-25T04:19:12.769541Z",
#                 "attributes": {
#                     "completion_tokens": 15,
#                     "prompt_tokens": 269,
#                     "total_tokens": 284,
#                     "finish_reason": "tool_calls",
#                     "finish_type": "success",
#                     "agentic_type": "agent_as_tool_call" ##TBD
#                 }
#             }
