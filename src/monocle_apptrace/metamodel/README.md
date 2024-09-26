# Monocle metamodel

## Overview 
Monocle metamodel is the way to manage standardization across all supported GenAI component stack. It includes the list of components that Monocle can identify and extract metadata. This help understanding and analyzing the traces from applications that include multiple components and can evolve over time. This is one of core value that Monocle provides to it's user community.

## Meta model
The Monocle metamodel comprises of three things, 
- Entity types, definitions of technology types and supported vendor implementations.
- A JSON format that overlays on top of Open Telemetry tracing format that includes the common attributes for each entity type.
- Map of component menthods to trace with instrumentation methods provided by Monocle.

### Entity type
The entity type defines the type of GenAI component that Monocle understand. The monocle instrumentation can extract the relevenat information for this entity. There are a fixed set of [entity types](./entity_types.py) that are defined by Monocle out of the box, eg workflow, model etc. As the GenAI landscape evolves, the Monocle community will introduce a new entity type if the current entities won't represent a new technology component.
Each entity types has number of supported technology components that Monocle handles out of the box, eg. LlamaIndex is a supported workflow. Monocle community will continue to expand the breadth of the project by adding more components.

### Span types
The GenAI application have specific [types of spans](./spans/README.md#span-types-and-events) where diffrent entities integrate. Monocle metamodel defines these types and specifies format for tracing data and metadata generated in such spans. 

### Consistent trace format
Monocle generates [traces](../../../Monocle_User_Guide.md#traces) which comprises of [spans](../../../Monocle_User_Guide.md#spans). Note that Monocle trace is [OpenTelemetry format](https://opentelemetry.io/docs/concepts/signals/traces/) compatible. Each span is essentially a step in the execution that interacts with one of more GenAI technology components. The please refer to the [full spec of the json format](./span_format.json) and a detailed [example](./span_example.json). 
The ```attribute``` section of the span includes a list of such entities that are used in that span.
The runtime data and metadata collected during the execution of that span are included in the ```events``` section of the trace (as per the Otel spec). Each entry in the event corrosponds to the entity involved in that trace execution if it has produced any runtime outputs. 
Please see the [span format](./spans/README.md) for details.

### Instrumentation method map
The map dectates what Monocle tracing method is relevant for the a given GenAI tech component method/API. It also specifies the name for that span to set in the trace output.
```python
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "span_name": "llamaindex.query",
        "wrapper_package": "wrap_common",
        "wrapper_method": "task_wrapper"
    }
```

## Extending the meta model
Monocle is highly extensible. This section describe when one would need to extend the meta model. Please refer to Monocle [User guide](../../../Monocle_User_Guide.md) and [Contributor guide](../../../Monocle_contributor_guide.md) for detailed steps.
### Trace a new method/API
If you have overloaded an existing functionality in one of the supported components by creating a new function. Monocle doesn't know that this function should be traced, say because it's calling an LLM. You could define a new mapping so Monocle instrumentation can trace this function the say way it handles other LLM invocation functions.

### Adding a new component provider
Let's say there's a new database that supports vector search capability which is not supported by the Monocle. In this case, first you'll need to add that database under the ``MonocleEntity.VectorDB`` list. Then you'll need to extend the method map and test if the existing Monocle tracing functions has logic to effectively trace the new component. If not, then you might need to implement new method to cover the gap and update the mapping table according.

### Support new type of entity
If there's new component that can't be mapped to any of the existing entity types, then it'll require extending the metamodel and implement new instrumetation to support it. We recommend you initiate a discussion with Monocle community to add the support.