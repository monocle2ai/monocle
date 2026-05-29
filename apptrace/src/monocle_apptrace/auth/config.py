"""Auth0 + Okahu endpoint resolution.

Defaults target prod. The Auth0 client_id, audience, and API hosts must come
from the same environment — mixing them produces a 401 "signing key not
found" because the API only trusts JWTs signed by its own Auth0 tenant.
Override via the MONOCLE_AUTH0_* / MONOCLE_OKAHU_* env vars to point at
stage.
"""
import os
from dataclasses import dataclass


"""Callback server for the Auth0 redirect."""
_DEFAULT_AUTH0_DOMAIN = "auth.okahu.co"
_DEFAULT_AUTH0_CLIENT_ID = "w5msUdJgpximQMqo8LqkrLyiQMYbYiDw"
_DEFAULT_AUTH0_AUDIENCE = "https://api.okahu.co/"
_DEFAULT_OKAHU_API_HOST = "api.okahu.co"
_DEFAULT_OKAHU_MGMT_HOST = "management.okahu.co"


@dataclass(frozen=True)
class AuthConfig:
    auth0_domain: str
    auth0_client_id: str
    auth0_audience: str
    okahu_api_host: str
    okahu_mgmt_host: str

    @property
    def authorize_url(self) -> str:
        return "https://{}/authorize".format(self.auth0_domain)

    @property
    def token_url(self) -> str:
        return "https://{}/oauth/token".format(self.auth0_domain)

    @property
    def tenant_url(self) -> str:
        return "https://{}/api/v1/tenant".format(self.okahu_api_host)

    @property
    def mgmt_tenant_url(self) -> str:
        return "https://{}/api/v1/tenant".format(self.okahu_mgmt_host)

    def keys_url(self, tenant_id: str) -> str:
        return "https://{}/api/v1/tenants/{}/keys".format(self.okahu_api_host, tenant_id)


def load_config() -> AuthConfig:
    return AuthConfig(
        auth0_domain=os.environ.get("MONOCLE_AUTH0_DOMAIN", _DEFAULT_AUTH0_DOMAIN),
        auth0_client_id=os.environ.get("MONOCLE_AUTH0_CLIENT_ID", _DEFAULT_AUTH0_CLIENT_ID),
        auth0_audience=os.environ.get("MONOCLE_AUTH0_AUDIENCE", _DEFAULT_AUTH0_AUDIENCE),
        okahu_api_host=os.environ.get("MONOCLE_OKAHU_API_HOST", _DEFAULT_OKAHU_API_HOST),
        okahu_mgmt_host=os.environ.get("MONOCLE_OKAHU_MGMT_HOST", _DEFAULT_OKAHU_MGMT_HOST),
    )
