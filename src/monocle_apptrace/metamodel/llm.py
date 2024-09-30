from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper




class LLM:
    def __init__(self, instance):
        self.instance = instance


    def add_llm_attributes(self, span):

        model_name = resolve_from_alias(self.instance.__dict__, ["model", "model_name"])

        span.set_attribute("entity.2.name", model_name)
        span.set_attribute("entity.2.type", "model.llm")
        span.set_attribute("entity.2.model_name",model_name)
