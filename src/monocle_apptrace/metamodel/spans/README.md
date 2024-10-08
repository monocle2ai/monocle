# Monocle Span format
Monocle generates [traces](../../../../Monocle_User_Guide.md#traces) which comprises of [spans](../../../../Monocle_User_Guide.md#spans). Note that Monocle trace is [OpenTelemetry format](https://opentelemetry.io/docs/concepts/signals/traces/) compatible. Each span is essentially a step in the execution that interacts with one of more GenAI technology components. This document explains the [span format](./span_format.json) that Monocle generates for GenAI application tracing.

Per the OpenTelemetry convention, each span contains an attribute section and event section. In Monocle generated trace, the attribute sections includes details of GenAI entities used in the span. The event section includes the input, output and metadata related to the execution of that span.

## Attributes 
The attribute sections includes details of GenAI entities used in the span. For each entity used in the span in includes the entity name and entity type. For every type of entity, there are required and optional attributes listed below.
### Json format
```json
    attributes: 
        "span.type": "Monocle-span-type",
        "entity.count": "count-of-entities",
    
        "entity.<index>.name": "Monocle-Entity-name",
        "entity.<index>.type": "MonocleEntity.<entity-type>"
        ...
```
The ```entity.count``` indicates total number of entities used in the given span. For each entity, the details are captured in ```entity.<index>.X```. For example, 
```json
    "attributes": {
        "span.type": "Inference",
        "entity.count": 2,
        "entity.1.name": "AzureOpenAI",
        "entity.1.type": "Inference.Azure_oai",
        "entity.2.name": "gpt-35-turbo",
        "entity.2.type": "Model.LLM",
        "entity.2.model_name": "gpt-35-turbo",
```

### Entity type specific attributes
#### MonocleEntity.Workflow
| Name | Description | Values | Required |
| - | - | - | - |
| name | Entity name generated by Monocle | Name String | Required |
| type | Monocle Entity type | MonocleEntity.Workflow | Required |
| optional-attribute | Additional attribute specific to entity |  | Optional |

### MonocleEntity.Model
| Name | Description | Values | Required |
| - | - | - | - |
| name | Entity name generated by Monocle | Name String | Required |
| type | Monocle Entity type | MonocleEntity.Model | Required |
| model_name | Name of model | String | Required |
| optional-attribute | Additional attribute specific to entity |  | Optional |

### MonocleEntity.AppHosting
| Name | Description | Values | Required |
| - | - | - | - |
| name | Entity name generated by Monocle | Name String | Required |
| type | Monocle Entity type | MonocleEntity.AppHosting | Required |
| optional-attribute | Additional attribute specific to entity |  | Optional |

### MonocleEntity.Inference
| Name | Description | Values | Required |
| - | - | - | - |
| name | Entity name generated by Monocle | Name String | Required |
| type | Monocle Entity type | MonocleEntity.Inference | Required |
| optional-attribute | Additional attribute specific to entity |  | Optional |

### MonocleEntity.VectorDB
| Name | Description | Values | Required |
| - | - | - | - |
| name | Entity name generated by Monocle | Name String | Required |
| type | Monocle Entity type | MonocleEntity.VectorDB | Required |
| optional-attribute | Additional attribute specific to entity |  | Optional |

## Events
The event section includes the input, output and metadata generated by that span execution. For each type of span, there are required and option input, output and metadata items listed below. If there's no data genearated in the space, the events will be an empty array.

### Json format
```json
    "events" : [
        {
            "name": "data.input",
            "timestamp": "UTC timestamp",
            "attributes": {
                "input_attribute": "value"
           }
        },
        {
            "name": "data.output",
            "timestamp": "UTC timestamp",
            "attributes": {
                "output_attribute": "value"
            }
        },
        {
            "name": "metadata",
            "timestamp": "UTC timestamp",
            "attributes": {
                "metadata_attribute": "value"
            }
        }
    ]
```

## Span types and events
The ```span.type``` captured in ```attributes``` section of the span dectates the format of the ```events```
### SpanType.Retrieval
| Name | Description | Values | Required |
| - | - | - | - |
| name | event name  | data.input or data.output or metadata | Required |
| timestamp | timestap when the event occurred | UTC timestamp | Required |
| attributes | input/output/metadata attributes generated in span | Dictionary | Required |

### SpanType.Inference
| Name | Description | Values | Required |
| - | - | - | - |
| name | event name  | data.input or data.output or metadata | Required |
| timestamp | timestap when the event occurred | UTC timestamp | Required |
| attributes | input/output/metadata attributes generated in span | Dictionary | Required |

### SpanType.Workflow
| Name | Description | Values | Required |
| - | - | - | - |
| name | event name  | data.input or data.output or metadata | Required |
| timestamp | timestap when the event occurred | UTC timestamp | Required |
| attributes | input/output/metadata attributes generated in span | Dictionary | Required |

### SpanType.Internal
Events will be empty