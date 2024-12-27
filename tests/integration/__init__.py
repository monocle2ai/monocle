import os

os.environ.setdefault("AZURE_OPENAI_API_DEPLOYMENT", "")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("MONOCLE_BLOB_CONNECTION_STRING", "")
os.environ.setdefault("MONOCLE_BLOB_CONTAINER_NAME", "")
os.environ.setdefault("OPENSEARCH_ENDPOINT_URL_BOTO", "")
os.environ.setdefault("OPENSEARCH_ENDPOINT_URL", "")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "")
os.environ.setdefault("AWS_ACCESS_KEY_ID_EXPORTER", "") # for aws exporter
os.environ.setdefault("AWS_SECRET_ACCESS_KEY_EXPORTER", "") # for aws exporter
os.environ.setdefault("MISTRAL_API_KEY", "")
os.environ.setdefault('MONOCLE_EXPORTER', "") #comma seperated values like "s3,blob,console"
os.environ.setdefault("MONOCLE_S3_BUCKET_NAME", "")
