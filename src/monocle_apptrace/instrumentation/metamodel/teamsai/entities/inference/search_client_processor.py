from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

SEARCH_CLIENT_PROCESSOR = {
    "type": "search",
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment",
                "attribute": "type",
                "accessor": lambda arguments: "azure.search"
            },
            {   "attribute": "version_remove_me",
                "accessor": lambda arguments: "11"
            },
            {
                "attribute": "index_name",
                "accessor": lambda arguments: _helper.extract_index_name(arguments['instance'])
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.extract_search_endpoint(arguments['instance'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "parameters",
                    "accessor":  lambda arguments: _helper.search_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "summary",
                    "accessor":  lambda arguments: _helper.search_output(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "attribute": "info",
                    "accessor": lambda arguments: _helper.capture_metadata(arguments)
                },
            ]
        }
    ]
}