from monocle_apptrace.langchain.metamodel.attributes import atttribute_util
from monocle_apptrace.utils import resolve_from_alias

inference ={
  "type": "inference",
  "attributes": [
    [
      {
        "_comment": "provider type ,name , deployment , inference_endpoint",
        "attribute": "type",
        "accessor": lambda arguments:'inference.azure_oai'
      },
      {
        "attribute": "provider_name",
        "accessor": lambda arguments:arguments['kwargs']['provider_name']
      },
      {
        "attribute": "deployment",
        "accessor": lambda arguments: resolve_from_alias(arguments['instance'].__dict__, ['engine', 'azure_deployment', 'deployment_name', 'deployment_id', 'deployment'])
      },
      {
        "attribute": "inference_endpoint",
        "accessor": lambda arguments: resolve_from_alias(arguments['instance'].__dict__, ['azure_endpoint', 'api_base']) or arguments['kwargs']['inference_endpoint']
      }
    ],
    [
      {
        "_comment": "LLM Model",
        "attribute": "name",
        "accessor": lambda arguments: resolve_from_alias(arguments['instance'].__dict__, ['model', 'model_name'])
      },
      {
        "attribute": "type",
        "accessor": lambda arguments: 'model.llm.'+resolve_from_alias(arguments['instance'].__dict__, ['model', 'model_name'])
      }
    ]
  ],
  "events": [
    { "name":"data.input",
      "attributes": [

        {
            "_comment": "this is instruction to LLM",
            "attribute": "system",
            "accessor": lambda arguments: atttribute_util.extract_messages(arguments)[0]
        },
        {
            "_comment": "this is user query to LLM",
            "attribute": "user",
            "accessor": lambda arguments: atttribute_util.extract_messages(arguments)[1]
        }
      ]
    },
    {
      "name":"data.output",
      "attributes": [
        {
            "_comment": "this is result from LLM",
            "attribute": "assistant",
            "accessor": lambda result: atttribute_util.extract_assistant_message(result)
        }
      ]
   }
  ]
}
