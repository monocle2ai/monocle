from monocle_apptrace.metamodel.llm import LLM
from monocle_apptrace.metamodel.provider import ProviderAttributes
class Inference:
    def __init__(self, instance):
        self.instance = instance


    def add_entities(self,span):
        span.set_attribute("span_type","Inference")
        span.set_attribute("entities_count",2)
        # name ,type, provider name,deployment , inference endpoint
        provider=ProviderAttributes(self.instance)
        provider.add_provider_attributes(span)
        #  name , type and model_type
        llm = LLM(self.instance)
        llm.add_llm_attributes(span)


