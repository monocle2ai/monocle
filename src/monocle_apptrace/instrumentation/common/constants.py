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
AZURE_FUNCTION_NAME = "azure.func"
AZURE_APP_SERVICE_NAME = "azure.asp"
AZURE_ML_SERVICE_NAME = "azure.mlw"
AWS_LAMBDA_SERVICE_NAME = "aws.lambda"
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
}

MONOCLE_INSTRUMENTOR = "monocle_apptrace"
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
WORKFLOW_TYPE_GENERIC = "workflow.generic"
MONOCLE_SDK_VERSION = "monocle_apptrace.version"