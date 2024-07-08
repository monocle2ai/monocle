const {
    WORKFLOW_TYPE_KEY,
    CONTEXT_INPUT_KEY,
    CONTEXT_OUTPUT_KEY,
    PROMPT_INPUT_KEY,
    PROMPT_OUTPUT_KEY,
    CONTEXT_PROPERTIES_KEY,
    LLM_INPUT_KEY,
    LLM_OUT_PUT_KEY,
    WORKFLOW_TYPE_MAP
} = require("../common/constants")

const langchainPackages = [
    {
        packagePath: "@langchain/core/language_models/chat_models",
        className: "BaseChatModel",
        methodName: "invoke",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            if (returnedValue instanceof Promise) {
                returnedValue.then(response => {
                    if (typeof response.content == "string") {
                        span.setAttribute(LLM_INPUT_KEY, response.content)
                    }
                })
            }
            const spanName = "langchain.ChatModel." + classInstance.getName()
            // langchain.task.ChatOpenAI
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/runnables",
        className: "RunnableParallel",
        methodName: "invoke",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const spanName = "langchain.workflow"
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/runnables",
        className: "RunnableSequence",
        methodName: "invoke",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const isRoot = typeof span.parentSpanId !== "string"
            if (isRoot) {

                const workflowName = span.resource.attributes.SERVICE_NAME
                span.setAttribute("workflow_name", workflowName)
                const workflow_type = WORKFLOW_TYPE_MAP.langchain
                span.setAttribute(WORKFLOW_TYPE_KEY, workflow_type)
                const prompt_input = arguments[0]
                if (typeof prompt_input == "string") {
                    span.setAttribute(PROMPT_INPUT_KEY, prompt_input)
                }

                const prompt_output = returnedValue
                if (typeof prompt_output == "string") {
                    span.setAttribute(PROMPT_OUTPUT_KEY, prompt_output)
                }
            }

            const spanName = "langchain.workflow"
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/vectorstores",
        className: "VectorStoreRetriever",
        methodName: "_getRelevantDocuments",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const context_input = arguments[0]
            if (typeof context_input == "string") {
                span.setAttribute(CONTEXT_INPUT_KEY, context_input)
            }

            let context_output = "";
            if (returnedValue instanceof Array) {
                returnedValue.forEach(val => {
                    if (typeof val.pageContent == "string") {
                        context_output += val.pageContent
                    }
                })
            }
            if (context_output.length > 0) {
                span.setAttribute(CONTEXT_OUTPUT_KEY, context_output)
            }

            const spanName = "langchain.task.VectorStoreRetriever"
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/prompts",
        className: "BaseChatPromptTemplate",
        methodName: "invoke",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const spanName = "langchain.task.ChatPromptTemplate"
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/prompts",
        className: "PromptTemplate",
        methodName: "format",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const spanName = "langchain.task.PromptTemplate"
            span.updateName(spanName)
        }
    }
]

exports.langchainPackages = langchainPackages