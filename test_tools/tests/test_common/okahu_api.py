import requests
import os
import dotenv
dotenv.load_dotenv()

OKAHU_API_BASE_URL = os.getenv("OKAHU_API_BASE_URL", "https://api.okahu.co")
OAKHU_PORTAL_BASE_URL = os.getenv("OKAHU_PORTAL_BASE_URL", "https://portal.okahu.co")
OKAHU_TRACE_VIEW_URL_TEMPLATE = OAKHU_PORTAL_BASE_URL + "/en/apps/{app_name}/traces?factName=traces&breakdownFilter=traces.genai"
CREATE_WORKFLOW_ENDPOINT = "v1/componets"
class OkahuAPI:
    def __init__(self, api_key: str=None, api_url: str = OKAHU_API_BASE_URL):
        self.api_url = api_url + "/api"
        self.api_key = api_key or os.getenv("OKAHU_API_KEY")
        if not self.api_key:
            raise ValueError("OKAHU_API_KEY not set. Please set the OKAHU_API_KEY environment variable.")
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def create_workflow(self, name: str, description: str) -> dict:
        """
        Create a new workflow in the Okahu API.

        Args:
            name (str): The name of the workflow.
            description (str): The description of the workflow.

        Returns:
            dict: The created workflow details.
        """
        url = f"{self.api_url}/v1/components/{name.replace(' ', '_').lower()}"
        headers = self.headers
        payload = {
            "display_name": name,
            "description": description,
            "type": "workflow.generic",
            "domain": "logical",
            "status": "active"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def create_app(self, app_name:str, description:str, components:list[str]) -> dict:
        """
        Create a new app in the Okahu API.

        Args:
            app_name (str): The name of the app.
            description (str): The description of the app.
        Returns:
            dict: The created app details.
        """
        url = f"{self.api_url}/v1/apps/{app_name.replace(' ', '_').lower()}"
        headers = self.headers
        payload = {
            "display_name": app_name,
            "description": description,
            "status": "active",
            "components": [{"component_name": comp} for comp in components]
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def app_discover(self, app_name:str) -> dict:
        """
        Trigger discovery for the app in the Okahu API.

        Returns:
            dict: The discovery response details.
        """
        url = f"{self.api_url}/v1/discover/architecture/apps/{app_name.replace(' ', '_').lower()}"
        headers = self.headers

        response = requests.put(url, headers=headers)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    api_key = os.getenv("OKAHU_API_KEY")
    okahu_api = OkahuAPI(api_key=api_key)
    workflow = okahu_api.create_workflow(name="Test Workflow 3", description="A test workflow created via API")
    print("Created Workflow:", workflow)
    app = okahu_api.create_app(app_name="Test app 3", description="A test app created via API", components=["test_workflow_3"])
    print("Created App:", app)