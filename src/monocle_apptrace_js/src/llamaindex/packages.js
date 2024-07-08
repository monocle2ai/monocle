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

const llamaindexPackages = [
    {
        packagePath: "llamaindex",
        className: "VectorIndexRetriever",
        methodName: "retrieve",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {

            const spanName = "llamaindex.retriever"
            span.updateName(spanName)

            const context_input = arguments[0].query
            if (typeof context_input == "string") {
                span.setAttribute(CONTEXT_INPUT_KEY, context_input)
            }

            const prompt_output = returnedValue
            if (typeof prompt_output == "string") {
                span.setAttribute(PROMPT_OUTPUT_KEY, prompt_output)
            }

            let context_output = "";
            if (returnedValue instanceof Array) {
                returnedValue.forEach(val => {
                    if (typeof val.node.text == "string") {
                        context_output += val.node.text
                    }
                })
            }
            if (context_output.length > 0) {
                span.setAttribute(CONTEXT_OUTPUT_KEY, context_output)
            }
        }
    },
    {
        packagePath: "llamaindex",
        className: "RetrieverQueryEngine",
        methodName: "query",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const isRoot = typeof span.parentSpanId !== "string"
            if (isRoot) {

                const workflowName = span.resource.attributes.SERVICE_NAME
                span.setAttribute("workflow_name", workflowName)
                const workflow_type = WORKFLOW_TYPE_MAP.llama_index
                span.setAttribute(WORKFLOW_TYPE_KEY, workflow_type)
                const prompt_input = arguments[0].query
                if (typeof prompt_input == "string") {
                    span.setAttribute(PROMPT_INPUT_KEY, prompt_input)
                }

                const prompt_output = returnedValue.message.content
                if (typeof prompt_output == "string") {
                    span.setAttribute(PROMPT_OUTPUT_KEY, prompt_output)
                }
            }

            const spanName = "llamaindex.queryengine"
            span.updateName(spanName)
        }
    },
    {
        packagePath: "llamaindex",
        className: "OpenAI",
        methodName: "chat",
        attributeSetter: ({ returnedValue, arguments, classInstance, span }) => {
            const spanName = "llamaindex.openai"
            span.updateName(spanName)
        }
    },

]

exports.llamaindexPackages = llamaindexPackages 