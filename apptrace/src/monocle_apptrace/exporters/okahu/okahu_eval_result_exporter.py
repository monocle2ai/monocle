import logging
import os
from typing import Optional, Dict, Any
import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)

OKAHU_PROD_EVALUATION_ENDPOINT = "https://eval.okahu.co/api/v1/eval/jobs"


class OkahuEvalResultExporter:
    """Exporter for sending evaluation results back to Okahu."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize the Okahu evaluation result exporter.
        
        Args:
            api_key: Okahu API key. If not provided, reads from OKAHU_API_KEY environment variable.
            base_url: Base URL for evaluation endpoint. If not provided, uses default or OKAHU_EVALUATION_ENDPOINT env var.
            timeout: Request timeout in seconds. Defaults to 15.
        """
        self.api_key = api_key or os.getenv("OKAHU_API_KEY")
        if not self.api_key:
            raise ValueError("OKAHU_API_KEY not set. Provide api_key or set environment variable.")
        
        self.base_url = (base_url or os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT)).rstrip("/")
        self.timeout = timeout or 15
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        })
        
        self._closed = False
    
    def export_results(
        self,
        job_id: str,
        eval_result: Any,
        template_name: str
    ) -> Dict[str, Any]:
        """
        Export evaluation results to Okahu.
        
        Args:
            job_id: The evaluation job ID from Okahu.
            eval_result: The evaluation results to export.
            template_name: The name of the evaluation template.
            
        Returns:
            Response data from Okahu.
            
        Raises:
            ValueError: If exporter is closed or parameters are invalid.
            AssertionError: If the request fails.
        """
        if self._closed:
            raise ValueError("Exporter is closed. Cannot export results.")
        
        if not job_id:
            raise ValueError("job_id is required.")
        if not template_name:
            raise ValueError("template_name is required.")
        
        url = f"{self.base_url}/v1/eval/jobs/{job_id}/results"
        payload = {
            "evaluation_results": eval_result,
            "template_name": template_name
        }
        
        try:
            response = self.session.post(
                url=url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.debug(
                "Evaluation results successfully exported to Okahu. Job ID: %s, Template: %s",
                job_id,
                template_name
            )
            
            return response.json()
            
        except Timeout as exc:
            error_msg = f"Evaluation result export timed out: {exc}"
            logger.error(error_msg)
            raise AssertionError(error_msg) from exc
        except RequestException as exc:
            error_msg = f"Failed to export evaluation results. Status: {getattr(exc.response, 'status_code', 'N/A')}, Error: {exc}"
            logger.error(error_msg)
            if hasattr(exc, 'response') and exc.response is not None:
                logger.error("Response: %s", exc.response.text)
            raise AssertionError(error_msg) from exc
    
    def delete_trace(self, trace_id: str, ingest_endpoint: Optional[str] = None) -> bool:
        """
        Delete a trace from Okahu evaluation storage.
        
        Args:
            trace_id: The trace ID to delete.
            ingest_endpoint: Optional ingest endpoint. If not provided, derives from env or default.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        if self._closed:
            logger.warning("Exporter is closed. Cannot delete trace.")
            return False
        
        if not trace_id:
            raise ValueError("trace_id is required.")
        
        ingest = ingest_endpoint or os.getenv("OKAHU_INGESTION_ENDPOINT", "https://ingest.okahu.co/api/v1/trace/ingest")
        delete_url = ingest.rstrip("/").replace("/trace/ingest", "/eval/delete")
        params = {"trace_id": trace_id}
        
        try:
            response = self.session.delete(
                url=delete_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug("Trace %s successfully deleted from Okahu.", trace_id)
            return True
            
        except RequestException as e:
            logger.debug("Failed to delete trace %s: %s", trace_id, e)
            if hasattr(e, 'response') and e.response is not None:
                logger.debug("Status Code: %s, Response: %s", e.response.status_code, e.response.text)
            return False
    
    def shutdown(self) -> None:
        """Close the session and mark exporter as closed."""
        if self._closed:
            logger.warning("Exporter already closed.")
            return
        
        if hasattr(self, 'session'):
            self.session.close()
        
        self._closed = True
        logger.debug("OkahuEvalResultExporter shutdown complete.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
