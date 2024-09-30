from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper

class EmbedModel:
    def __init__(self, instance):
        self.instance = instance

    def add_embedmodel_attributes(self, span ,embedding_model):
        # Resolve model_name using the alias list
        embed_model = self.instance.__dict__.get('_embed_model').__class__.__name__

        # Fallback to None or specific default values if not found
        # temperature = self.instance.__dict__.get("temperature", None)

        # Check if model_name is valid and set the attributes accordingly
        if embed_model:
            span.set_attribute("entity.2.name", embedding_model)
            span.set_attribute("entity.2.type", "model.embedding")
            span.set_attribute("entity.2.model_name", embedding_model)
        # else:
        #     span.set_attribute("entity.2.name", "Unknown")
        #     span.set_attribute("entity.2.type", "Unknown")

        # Add temperature if it exists
        # if temperature is not None:
        #     span.set_attribute("entity.1.temperature", temperature)