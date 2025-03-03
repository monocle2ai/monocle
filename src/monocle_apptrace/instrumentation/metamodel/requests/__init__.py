from os import environ
allowed_urls = environ.get('MONOCLE_TRACE_PROPAGATATION_URLS', ' ').split(',')