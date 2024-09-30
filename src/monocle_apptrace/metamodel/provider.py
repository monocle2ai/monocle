from urllib.parse import urlparse
from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper


class ProviderAttributes:
    def __init__(self, instance):
        self.instance = instance
    def set_provider_name(self):
        provider_url = ""

        try:
            if isinstance(self.instance.client._client.base_url.host, str):
                provider_url = self.instance.client._client.base_url.host
        except:
            pass

        try:
            if isinstance(self.instance.api_base, str):
                provider_url = self.instance.api_base
        except:
            pass

        try:
            if len(provider_url) > 0:
                parsed_provider_url = urlparse(provider_url)
                # curr_span.set_attribute("provider_name", parsed_provider_url.hostname or provider_url)
                return parsed_provider_url.hostname or provider_url
        except:
            pass
    def add_provider_attributes(self, span):
        # Assuming the provider details are in instance.provider
        provider_name = ""

        # Handle provider URL if available
        try:
            provider_name = self.set_provider_name()
        except AttributeError:
            pass
        # Handle inference endpoint if available
        inference_ep = resolve_from_alias(self.instance.__dict__, ["azure_endpoint", "api_base"])
        deployment_name = resolve_from_alias(self.instance.__dict__, ["engine", "azure_deployment",
                                                                      "deployment_name", "deployment_id", "deployment"])
        span.set_attribute("entity.1.name","AzureOpenAI")
        span.set_attribute("entity.1.type", "inference.azure_oai")
        span.set_attribute("entity.1.provider_name", provider_name)
        span.set_attribute("entity.1.deployment",deployment_name)
        span.set_attribute("entity.1.inference_endpoint",inference_ep)

