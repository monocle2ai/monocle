INFERENCE = {
  "type": "inference",
  "attributes": [
    [
      {
        "_comment": "provider type ,name , deployment , inference_endpoint",
        "attribute": "meta_type",
        "accessor": lambda arguments:'inference.azure_oai'
      },
      {
        "attribute": "meta_provider_name",
        "accessor": lambda arguments: 'dummy_provider_name'
      },
      {
        "attribute": "meta_deployment",
        "accessor": lambda arguments: 'dummy_deployment'
      },
      {
        "attribute": "meta_inference_endpoint",
        "accessor": lambda arguments: 'dummy_endpoint'
      }
    ],
    [
      {
        "_comment": "LLM Model",
        "attribute": "meta_name",
        "accessor": lambda arguments: 'dummy_name'
      },
      {
        "attribute": "meta_type",
        "accessor": lambda arguments: 'dummy_type'
      }
    ]
  ],
  "events": [
    { "name":"data.input",
      "attributes": [

          {
              "_comment": "this is instruction and user query to LLM",
              "attribute": "meta_input",
              "accessor": lambda arguments: 'dummy_input'
          }
      ]
    },
    {
      "name":"data.output",
      "attributes": [
        {
            "_comment": "this is result from LLM",
            "attribute": "meta_response",
            "accessor": lambda arguments: 'dummy_response'
        }
      ]
   }
  ]
}