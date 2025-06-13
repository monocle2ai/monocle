from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

AZUREAISEARCH_OUTPUT_PROCESSOR = {
    "type": "generic",
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment",
                "attribute": "type",
                "accessor": lambda arguments: 'teams.AzureAISearch'
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
            "_comment": "input to Teams AI",
            "attributes": [
                {
                    "_comment": "search input",
                    "accessor":  lambda arguments: _helper.search_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "output from Teams AI",
            "attributes": [
                {
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.get_status(arguments)
                },
                {
                    "attribute": "status_code",
                    "accessor": lambda arguments: _helper.get_status_code(arguments)
                },
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.get_response(arguments)
                },
                {
                    "attribute": "check_status",
                    "accessor": lambda arguments: _helper.check_status(arguments)
                }
            ]
        },
    ]
}