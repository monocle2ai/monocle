from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

SEARCH_POST_PROCESSOR = {
    "type": "search",
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "search_text",
                    "accessor":  lambda arguments: _helper.search_post_input(arguments['kwargs'])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "results",
                    "accessor":  lambda arguments: _helper.search_post_output(arguments['result'])
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "attribute": "options",
                    "accessor": lambda arguments: _helper.search_post_capture_meta(arguments)
                },
            ]
        }
    ]
}