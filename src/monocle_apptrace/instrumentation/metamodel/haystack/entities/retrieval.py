from monocle_apptrace.instrumentation.metamodel.haystack import (_helper, )
from monocle_apptrace.instrumentation.common.utils import get_attribute

RETRIEVAL = {
    "type": "retrieval",
    "attributes": [
        [
            {
                "_comment": "vector store name and type",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['instance'].__dict__,
                                                                         ['document_store',
                                                                          '_document_store']).__class__.__name__
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'vectorstore.' + _helper.resolve_from_alias(
                    arguments['instance'].__dict__, ['document_store', '_document_store']).__class__.__name__
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: _helper.get_vectorstore_deployment(
                    _helper.resolve_from_alias(arguments['instance'].__dict__,
                                               ['document_store', '_document_store']).__dict__)
            }
        ],
        [
            {
                "_comment": "embedding model name and type",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_embeding_model(arguments['instance'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + _helper.extract_embeding_model(arguments['instance'])
            }
        ]
    ],
    "events": [
        {"name": "data.input",
         "attributes": [

             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: get_attribute("input")
             }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.update_output_span_events(arguments['result'])
                }
            ]
        }
    ]
}
