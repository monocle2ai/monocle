from monocle_apptrace.providers.base import Provider


class OkahuProvider(Provider):
    name = "okahu"
    span_exporter = {"module": "monocle_apptrace.providers.okahu.okahu_exporter", "class": "OkahuSpanExporter"}

    def get_setup_env(self, args):
        # Imported here so `monocle-apptrace --version` stays fast.
        from monocle_apptrace.providers.okahu.setup import get_setup_env
        return get_setup_env(args)
