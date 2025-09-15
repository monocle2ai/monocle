from os import environ
from monocle_apptrace.instrumentation.common.constants import TRACE_PROPOGATION_URLS
allowed_url_str = environ.get(TRACE_PROPOGATION_URLS, "")
allowed_urls:list[str] = [] if allowed_url_str == "" else allowed_url_str.split(',')