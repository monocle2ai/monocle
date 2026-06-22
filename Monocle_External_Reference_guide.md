# Monocle External Reference Guide

This guide provides information about external services and dependencies used by Monocle for exporting trace data.

## Exporters

Monocle supports multiple export targets for trace data. Each exporter requires specific dependencies and configuration.

### AWS S3 Exporter

The S3 exporter allows you to export trace data to Amazon S3 buckets.

#### Prerequisites

- **AWS Account**: An active AWS account with S3 access
- **AWS Credentials**: Access key ID and secret access key with S3 write permissions
- **S3 Bucket**: An existing S3 bucket or permissions to create one


#### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONOCLE_AWS_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID` | AWS access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `MONOCLE_AWS_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY` | AWS secret access key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `MONOCLE_S3_BUCKET_NAME` | S3 bucket name for storing traces | `my-monocle-traces` |
| `MONOCLE_S3_FILE_PREFIX` | (Optional) Prefix for trace files | `monocle_trace_` |

#### Configuration

Set the exporter in your environment:
```bash
export MONOCLE_EXPORTER=s3
export MONOCLE_S3_BUCKET_NAME=my-traces-bucket
export MONOCLE_AWS_ACCESS_KEY_ID=your-access-key
export MONOCLE_AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Reference

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

---

### Azure Blob Storage Exporter

The Blob exporter allows you to export trace data to Azure Blob Storage containers.

#### Prerequisites

- **Azure Account**: An active Azure account with Storage access
- **Azure Storage Account**: A storage account with appropriate permissions
- **Container**: An existing container or permissions to create one

#### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONOCLE_BLOB_CONNECTION_STRING` | Azure Storage connection string | `DefaultEndpointsProtocol=https;AccountName=...` |
| `MONOCLE_BLOB_CONTAINER_NAME` | Container name for storing traces | `monocle-traces` |
| `MONOCLE_BLOB_FILE_PREFIX` | (Optional) Prefix for trace files | `monocle_trace_` |

#### Configuration

Set the exporter in your environment:
```bash
export MONOCLE_EXPORTER=blob
export MONOCLE_BLOB_CONNECTION_STRING="your-connection-string"
export MONOCLE_BLOB_CONTAINER_NAME=monocle-traces
```

#### Reference

- [Azure Blob Storage Documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/)

---

### Google Cloud Storage (GCS) Exporter

The GCS exporter allows you to export trace data to Google Cloud Storage buckets.

#### Prerequisites

- **Google Cloud Account**: An active GCP account with Cloud Storage access
- **GCS Bucket**: An existing bucket or permissions to create one
- **Authentication**: Service account credentials or application default credentials

#### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONOCLE_GCS_BUCKET_NAME` | GCS bucket name for storing traces | `my-monocle-traces` |
| `MONOCLE_GCS_PROJECT_ID` | (Optional) GCP project ID | `my-project-id` |
| `MONOCLE_GCS_LOCATION` | (Optional) Bucket location | `US` (default) |
| `MONOCLE_GCS_FILE_PREFIX` | (Optional) Prefix for trace files | `monocle_trace_` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key file | `/path/to/service-account.json` |

#### Configuration

Set the exporter in your environment:
```bash
export MONOCLE_EXPORTER=gcs
export MONOCLE_GCS_BUCKET_NAME=my-traces-bucket
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

#### Reference

- [Google Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Authentication Guide](https://cloud.google.com/docs/authentication/getting-started)

---

### Okahu Cloud Platform

Okahu is a cloud-based observability platform for AI applications that provides trace ingestion, analysis, and evaluation capabilities.

#### Prerequisites

- **Okahu Account**: Sign up at [okahu.co](https://okahu.co)
- **API Key**: Generate an API key from your Okahu dashboard

#### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OKAHU_API_KEY` | Okahu API key for authentication | `okh_1234567890abcdef` |
| `OKAHU_INGESTION_ENDPOINT` | (Optional) Custom ingestion endpoint | `https://ingest.okahu.co/api/v1/trace/ingest` |

#### Configuration

Set the exporter in your environment:
```bash
export MONOCLE_EXPORTER=okahu
export OKAHU_API_KEY=your-okahu-api-key
```

#### Reference

- [Okahu Documentation](https://docs.okahu.co)
- [Okahu API Reference](https://docs.okahu.co/api)

---

### Paygentic Usage Tracking

Paygentic is a usage-based billing and metering platform for AI applications that tracks token usage, API calls, and costs.

#### Prerequisites

- **Paygentic Account**: Sign up at [paygentic.io](https://paygentic.io)
- **API Key**: Generate an API key from your Paygentic dashboard
- **Subscription/Customer IDs**: Properly configured scope attributes

#### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PAYGENTIC_API_KEY` | Paygentic API key for authentication | `pg_1234567890abcdef` |
| `PAYGENTIC_ENDPOINT` | (Optional) API endpoint | `https://api.paygentic.io/v0/events` (production) |

#### Required Scope Attributes

Paygentic requires subscription and customer identifiers to be set as scope attributes:

```python
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope

# Set customer and subscription context
start_scope("subscriptionId", "sub_123456")
start_scope("customerId", "cust_789012")

# Your instrumented code here
# ...

# Clean up scopes when done
stop_scope("subscriptionId")
stop_scope("customerId")
```

#### Configuration

Set the exporter in your environment:
```bash
export MONOCLE_EXPORTER=paygentic
export PAYGENTIC_API_KEY=your-paygentic-api-key
```

#### Features

- Automatic token usage tracking from inference spans
- CloudEvents-based event streaming
- Subscription and customer-level metering
- Multi-provider cost aggregation (OpenAI, Anthropic, Google, AWS, Azure)
- Real-time usage dashboards

#### Reference

- [Paygentic Documentation](https://docs.paygentic.io)
- [Monocle Integration Guide](https://docs.paygentic.io/integrations/monocle)

---

### Open Telemetry exporter
#### Reference
- [OpenTelemetry Exporters](https://opentelemetry.io/docs/instrumentation/python/exporters/)


## Open Telemetry
Monocle spans are Open Telemetry spec compatible.
#### Reference
- [Otel span format](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/overview.md#spans)