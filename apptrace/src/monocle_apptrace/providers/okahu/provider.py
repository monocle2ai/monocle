from monocle_apptrace.providers.base import Provider


class OkahuProvider(Provider):
    name = "okahu"
    span_exporter = {"module": "monocle_apptrace.providers.okahu.okahu_exporter", "class": "OkahuSpanExporter"}
