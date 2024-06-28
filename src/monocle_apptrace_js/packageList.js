
const WORKFLOW_TYPE_KEY = "workflow_type"
const CONTEXT_INPUT_KEY = "workflow_context_input"
const CONTEXT_OUTPUT_KEY = "workflow_context_output"
const PROMPT_INPUT_KEY = "workflow_input"
const PROMPT_OUTPUT_KEY = "workflow_output"
const CONTEXT_PROPERTIES_KEY = "workflow_context_properties"
const LLM_INPUT_KEY = "llm_input"
const LLM_OUT_PUT_KEY = "llm_output"

const packageList_langchain = [
    {
        packagePath: "@langchain/core/language_models/chat_models",
        className: "BaseChatModel",
        methodName: "invoke",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            if (returnedValue instanceof Promise){
                returnedValue.then(response=>{
                    if (typeof response.content == "string"){
                        span.setAttribute(LLM_INPUT_KEY, response.content)
                    }
                })
            }
            const spanName = "langchain.ChatModel"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/runnables",
        className: "RunnableParallel",
        methodName: "invoke",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            if (returnedValue instanceof Promise){
                returnedValue.then(response=>{
                    console.log(response)
                })
            }
            const spanName = "langchain.workflow"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    
    {
        packagePath: "@langchain/core/runnables",
        className: "RunnableSequence",
        methodName: "invoke",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            if (returnedValue instanceof Promise){
                returnedValue.then(response=>{
                    console.log(response)
                })
            }
            const spanName = "langchain.workflow"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/vectorstores",
        className: "VectorStoreRetriever",
        methodName: "_getRelevantDocuments",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "langchain.Retriever"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/prompts",
        className: "BaseChatPromptTemplate",
        methodName: "invoke",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "langchain.task.ChatPromptTemplate"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "@langchain/core/prompts",
        className: "PromptTemplate",
        methodName: "format",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "langchain.task.PromptTemplate"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    }
]

const packageList_llamaindex = [
    {
        packagePath: "llamaindex",
        className: "VectorIndexRetriever",
        methodName: "retrieve",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "llamaindex.retriever"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "llamaindex",
        className: "RetrieverQueryEngine",
        methodName: "query",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "llamaindex.queryengine"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    {
        packagePath: "llamaindex",
        className: "OpenAI",
        methodName: "chat",
        attributeSetter : ({returnedValue, arguments, classInstance, span}) => 
        {
            const spanName = "llamaindex.openai"
            console.log(`in attributeSetter: ${spanName}`)
            span.updateName(spanName)
        }
    },
    
]

exports.PACKAGE_LIST = packageList_langchain.concat(packageList_llamaindex) 