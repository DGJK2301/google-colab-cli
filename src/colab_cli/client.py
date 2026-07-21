# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
import uuid

from colab_cli.utils import get_status_code
from pydantic import BaseModel, Field, TypeAdapter
import requests

# Standard Colab Headers
ACCEPT_JSON_HEADER = {"key": "Accept", "value": "application/json"}
COLAB_CLIENT_AGENT_HEADER = {
    "key": "X-Colab-Client-Agent",
    "value": "colab-cli",
}
COLAB_XSRF_TOKEN_HEADER = {"key": "X-Goog-Colab-Token", "value": ""}
# Marks a request as one that should be resolved through the Colab tunnel
# (Tunnel Frontend). Required by TFE-intercepted paths such as the keep-alive
# ping; without it the front-door rejects the request with HTTP 400.
COLAB_TUNNEL_HEADER = {"key": "X-Colab-Tunnel", "value": "Google"}

# Per-request timeout (seconds) for the keep-alive tunnel ping. TFE records the
# activity as soon as the request arrives, so we do not need to wait long for
# the (often non-responding) VM. A short timeout keeps the keep-alive daemon
# responsive on its 60s cadence.
KEEP_ALIVE_TIMEOUT = 10


@dataclass
class ColabEnvironment(abc.ABC):
    domain: str
    api: str


@dataclass
class Prod(ColabEnvironment):
    domain: str = "https://colab.research.google.com"
    api: str = "https://colab.pa.googleapis.com"


def uuid_to_web_safe_base64(uuid_val: uuid.UUID) -> str:
    uuid_str = str(uuid_val)
    transformed = uuid_str.replace("-", "_")
    padding = "." * (44 - len(uuid_str))
    return transformed + padding


class Accelerator(str, Enum):
    NONE = "NONE"
    G4 = "G4"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    V5E1 = "V5E1"
    V6E1 = "V6E1"


class Variant(str, Enum):
    DEFAULT = "DEFAULT"
    GPU = "GPU"
    TPU = "TPU"


class AssignmentVariant(int, Enum):
    DEFAULT = 0
    GPU = 1
    TPU = 2


class Shape(int, Enum):
    STANDARD = 0
    HIGH_RAM = 1


class RuntimeProxyInfo(BaseModel):
    token: str
    token_expires_in_seconds: int = Field(..., alias="tokenExpiresInSeconds")
    url: str


class ListedAssignment(BaseModel):
    accelerator: Accelerator
    endpoint: str
    variant: AssignmentVariant
    machine_shape: Shape = Field(..., alias="machineShape")
    runtime_proxy_info: RuntimeProxyInfo = Field(..., alias="runtimeProxyInfo")


class ListedAssignments(BaseModel):
    assignments: List[ListedAssignment]


class PostAssignmentResponse(BaseModel):
    accelerator: Accelerator
    endpoint: str
    runtime_proxy_info: RuntimeProxyInfo = Field(..., alias="runtimeProxyInfo")
    variant: AssignmentVariant


class GetAssignmentResponse(BaseModel):
    acc: str = Field(..., alias="acc")
    nbh: str = Field(..., alias="nbh")
    token: str = Field(..., alias="token")
    variant: Variant = Field(..., alias="variant")


class GetUnassignRequest(BaseModel):
    token: str


class Assignment(BaseModel):
    endpoint: str
    runtime_proxy_info: RuntimeProxyInfo = Field(..., alias="runtimeProxyInfo")


XSSI_PREFIX = ")]}'\n"
TUN_ENDPOINT = "/tun/m"


class ColabRequestError(Exception):
    def __init__(self, message, request, response, response_body=None):
        super().__init__(message)
        self.request = request
        self.response = response
        self.response_body = response_body


class TooManyAssignmentsError(ColabRequestError):
    """Deprecated compatibility type for an ambiguous HTTP 412 response.

    The historical name is not a reliable diagnosis: HTTP 412 can represent
    usage limits, entitlement, capacity, or assignment-count constraints. The
    type remains raised for API compatibility, but now preserves the original
    request, response, and response body and is also a ``ColabRequestError``.
    """

    def __init__(self, message, request=None, response=None, response_body=None):
        super().__init__(message, request, response, response_body)


class Client:
    def __init__(self, env: ColabEnvironment, session, logger=None):
        self.colab_domain = env.domain
        self.colab_api_domain = env.api
        self.session = session
        self.logger = logger or logging.getLogger(__name__)

    def _strip_xssi_prefix(self, v: str) -> str:
        if not v.startswith(XSSI_PREFIX):
            return v
        return v[len(XSSI_PREFIX) :]

    def _issue_request(
        self,
        endpoint: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.hostname in urlparse(self.colab_domain).hostname:
            if params is None:
                params = {}
            params["authuser"] = "0"

        request_headers = headers.copy() if headers else {}
        request_headers[ACCEPT_JSON_HEADER["key"]] = ACCEPT_JSON_HEADER["value"]
        request_headers[COLAB_CLIENT_AGENT_HEADER["key"]] = COLAB_CLIENT_AGENT_HEADER[
            "value"
        ]

        self.logger.debug(f"Request: {method} {endpoint}")
        self.logger.debug(f"Params: {params}")

        response = self.session.request(
            method, endpoint, headers=request_headers, params=params, **kwargs
        )

        self.logger.debug(f"Request Headers: {response.request.headers}")
        self.logger.debug(f"Response: {response.status_code} {response.reason}")
        self.logger.debug(f"Response Headers: {response.headers}")
        self.logger.debug(f"Response Body: {response.text}")
        if not response.ok:
            raise ColabRequestError(
                f"Failed to issue request {method} {endpoint}: {response.reason}",
                request=response.request,
                response=response,
                response_body=response.text,
            )

        body = self._strip_xssi_prefix(response.text)
        if not body:
            return
        # Some endpoints (e.g. KeepAliveAssignment) return a non-empty body
        # but the caller doesn't care about the response content — skip
        # pydantic validation entirely when no schema was supplied.
        if schema is None:
            return
        return TypeAdapter(schema).validate_python(json.loads(body))

    def list_assignments(self) -> List[ListedAssignment]:
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/assignments")
        assignments = self._issue_request(url, schema=ListedAssignments)
        return assignments.assignments

    def unassign(self, endpoint: str):
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/unassign/{endpoint}")
        resp = self._retry_idempotent_request(
            lambda: self._issue_request(url, schema=GetUnassignRequest),
            description=f"unassign token GET for {endpoint}",
        )
        headers = {COLAB_XSRF_TOKEN_HEADER["key"]: resp.token}
        try:
            return self._issue_request(
                url, method="POST", headers=headers, schema=BaseModel
            )
        except requests.RequestException as error:
            if self._reconcile_ambiguous_unassignment(endpoint):
                self.logger.warning(
                    "Confirmed release of %s after the unassign POST response was lost",
                    endpoint,
                )
                return None
            error.add_note(
                "The unassign POST result was ambiguous and the endpoint is still "
                "present or could not be checked. Local session state must be kept "
                "so `colab stop` can retry safely."
            )
            raise

    def _retry_idempotent_request(self, operation, *, description: str):
        """Retry a control-plane read without changing POST semantics."""

        for attempt in range(3):
            try:
                return operation()
            except requests.RequestException as error:
                if attempt == 2:
                    raise
                self.logger.debug(
                    "%s attempt %d failed: %s",
                    description,
                    attempt + 1,
                    error,
                )
                time.sleep(0.5 * (attempt + 1))
        raise AssertionError("unreachable")

    def _reconcile_ambiguous_unassignment(self, endpoint: str) -> bool:
        """Confirm a release after losing the POST response, without replaying it."""

        for attempt in range(3):
            if attempt:
                time.sleep(0.5 * attempt)
            try:
                assignments = self.list_assignments()
            except requests.RequestException as error:
                self.logger.debug(
                    "Unassign reconciliation GET %d failed: %s",
                    attempt + 1,
                    error,
                )
                continue
            if all(assignment.endpoint != endpoint for assignment in assignments):
                return True
        return False

    def assign(
        self,
        notebook_hash: uuid.UUID,
        variant: Optional[Variant] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> Union[PostAssignmentResponse, Assignment]:
        assignment = self._get_assignment(notebook_hash, variant, accelerator)
        if isinstance(assignment, Assignment):
            return assignment

        try:
            return self._post_assignment(
                notebook_hash, assignment.token, variant, accelerator
            )
        except requests.RequestException as error:
            reconciled = self._reconcile_ambiguous_assignment(
                notebook_hash, variant, accelerator
            )
            if reconciled is not None:
                self.logger.warning(
                    "Recovered assignment %s after the POST response was lost",
                    reconciled.endpoint,
                )
                return reconciled
            error.add_note(
                "The assignment POST result was ambiguous and could not be "
                "reconciled with the same notebook hash. Run `colab sessions` "
                "and release any `[?]` endpoint with `colab stop --endpoint ENDPOINT`."
            )
            raise
        except ColabRequestError as error:
            if get_status_code(error) != 412:
                raise

            # Preserve the legacy exception contract without preserving its
            # misleading diagnosis. Command callers see the original HTTP
            # evidence through the ColabRequestError base class.
            raise TooManyAssignmentsError(
                str(error),
                request=error.request,
                response=error.response,
                response_body=error.response_body,
            ) from error

    def _reconcile_ambiguous_assignment(
        self,
        notebook_hash: uuid.UUID,
        variant: Optional[Variant],
        accelerator: Optional[Accelerator],
    ) -> Optional[Assignment]:
        """Resolve a lost POST response without replaying the POST request."""

        for attempt in range(3):
            if attempt:
                time.sleep(0.5 * attempt)
            try:
                result = self._get_assignment(notebook_hash, variant, accelerator)
            except requests.RequestException as error:
                self.logger.debug(
                    "Assignment reconciliation GET %d failed: %s",
                    attempt + 1,
                    error,
                )
                continue
            if isinstance(result, Assignment):
                return result
        return None

    def _build_assign_url(
        self,
        notebook_hash: uuid.UUID,
        variant: Optional[Variant] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> str:
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/assign")
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant:
            params["variant"] = variant.value
        if accelerator:
            params["accelerator"] = accelerator.value

        req = requests.Request("GET", url, params=params)
        prep = req.prepare()
        return prep.url

    def _get_assignment(
        self,
        notebook_hash: uuid.UUID,
        variant: Optional[Variant] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> Union[GetAssignmentResponse, Assignment]:
        url = self._build_assign_url(notebook_hash, variant, accelerator)
        return self._issue_request(url, schema=Union[GetAssignmentResponse, Assignment])

    def _post_assignment(
        self,
        notebook_hash: uuid.UUID,
        xsrf_token: str,
        variant: Optional[Variant] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> PostAssignmentResponse:
        url = self._build_assign_url(notebook_hash, variant, accelerator)
        headers = {COLAB_XSRF_TOKEN_HEADER["key"]: xsrf_token}
        return self._issue_request(
            url, method="POST", headers=headers, schema=PostAssignmentResponse
        )

    def keep_alive_assignment(self, endpoint: str):
        """Refreshes the idle timer for the given assignment endpoint.

        TFE notes the activity as soon as the request arrives, then forwards it
        to the VM, which does not always respond on this path — so the request
        commonly read-times-out even though the keep-alive succeeded. A read
        timeout is therefore treated as success; only an actual HTTP error
        response (4xx/5xx, e.g. 404 for a deleted assignment) is surfaced.
        """
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/{endpoint}/keep-alive/")
        headers = {COLAB_TUNNEL_HEADER["key"]: COLAB_TUNNEL_HEADER["value"]}
        try:
            return self._issue_request(
                url, method="GET", headers=headers, timeout=KEEP_ALIVE_TIMEOUT
            )
        except requests.exceptions.ReadTimeout:
            # The activity was recorded by TFE before the request was forwarded;
            # the VM simply didn't answer in time. This is the normal,
            # successful case for this path.
            return None
