# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

import base64
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests import Response

from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState


@pytest.fixture
def session():
    return SessionState(
        name="test-session",
        token="test-token",
        url="https://fake-endpoint.colab.dev",
        endpoint="endpoint",
    )


@pytest.fixture
def http_session():
    return MagicMock(spec=requests.Session)


@pytest.fixture
def client(session, http_session):
    return ContentsClient(
        session,
        http_session=http_session,
    )


def _response(*, status=200, payload=None):
    response = MagicMock(spec=Response)
    response.status_code = status
    response.json.return_value = payload or {}
    return response


def test_list_dir(client, http_session):
    response = _response(
        payload={
            "name": "content",
            "type": "directory",
            "content": [
                {
                    "name": "file.txt",
                    "type": "file",
                },
                {
                    "name": "dir",
                    "type": "directory",
                },
            ],
        }
    )
    http_session.request.return_value = response

    result = client.list_dir("content")

    http_session.request.assert_called_once_with(
        "GET",
        ("https://fake-endpoint.colab.dev/api/contents/content"),
        params={
            "authuser": "0",
            "colab-runtime-proxy-token": "test-token",
        },
        json=None,
        timeout=(10.0, 60.0),
    )
    response.close.assert_called_once_with()
    assert result["type"] == "directory"


def test_rm_file(client, http_session):
    response = _response(status=204)
    http_session.request.return_value = response

    client.rm("content/file.txt")

    http_session.request.assert_called_once_with(
        "DELETE",
        ("https://fake-endpoint.colab.dev/api/contents/content/file.txt"),
        params={
            "authuser": "0",
            "colab-runtime-proxy-token": "test-token",
        },
        json=None,
        timeout=(10.0, 60.0),
    )
    response.close.assert_called_once_with()


def test_404_error(client, http_session):
    response = _response(status=404)
    http_session.request.return_value = response

    with pytest.raises(FileNotFoundError):
        client.list_dir("nonexistent")

    response.close.assert_called_once_with()


def test_download_file(
    client,
    http_session,
    tmp_path,
):
    content_bytes = b"Hello world!"
    response = _response(
        payload={
            "name": "test.txt",
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(content_bytes).decode("ascii"),
        }
    )
    http_session.request.return_value = response
    local_file = tmp_path / "test.txt"

    client.download(
        "content/test.txt",
        str(local_file),
    )

    assert local_file.read_bytes() == content_bytes
    response.close.assert_called_once_with()


def test_upload_file(
    client,
    http_session,
    tmp_path,
):
    response = _response()
    http_session.request.return_value = response
    local_file = tmp_path / "test.txt"
    local_file.write_bytes(b"Hello upload!")

    client.upload(
        str(local_file),
        "content/test.txt",
    )

    payload = http_session.request.call_args.kwargs["json"]
    assert payload["content"] == base64.b64encode(b"Hello upload!").decode("ascii")
    assert payload["chunk"] == 1
    response.close.assert_called_once_with()


def test_upload_chunk_uses_large_file_manager_markers(
    client,
    http_session,
):
    response = _response(
        payload={
            "type": "file",
            "size": 3,
        }
    )
    http_session.request.return_value = response

    client.upload_chunk(
        "content/archive.part",
        b"abc",
        chunk=2,
    )

    payload = http_session.request.call_args.kwargs["json"]
    assert payload["content"] == base64.b64encode(b"abc").decode("ascii")
    assert payload["chunk"] == 2
    assert http_session.request.call_args.kwargs["timeout"] == (60.0, 120.0)


def test_upload_chunk_accepts_independent_write_timeout(
    session,
    http_session,
):
    response = _response(
        payload={
            "type": "file",
            "size": 3,
        }
    )
    http_session.request.return_value = response
    client = ContentsClient(
        session,
        request_timeout=(2.0, 3.0),
        upload_request_timeout=(11.0, 12.0),
        http_session=http_session,
    )

    client.upload_chunk(
        "content/archive.part",
        b"abc",
        chunk=1,
    )

    assert http_session.request.call_args.kwargs["timeout"] == (11.0, 12.0)


def test_owned_session_is_reused_and_closed(session):
    with patch("colab_cli.contents.requests.Session") as factory:
        response = _response(
            payload={
                "type": "directory",
                "content": [],
            }
        )
        factory.return_value.request.return_value = response
        client = ContentsClient(session)
        client.list_dir("content")
        client.list_dir("content")
        client.close()
        client.close()

    factory.assert_called_once_with()
    assert factory.return_value.request.call_count == 2
    factory.return_value.close.assert_called_once_with()


def test_context_manager_closes_owned_session(
    session,
):
    with patch("colab_cli.contents.requests.Session") as factory:
        with ContentsClient(session):
            pass

    factory.return_value.close.assert_called_once_with()


def test_injected_session_is_not_closed(
    client,
    http_session,
):
    client.close()
    http_session.close.assert_not_called()


def test_closed_client_refuses_new_requests(
    client,
    http_session,
):
    client.close()

    with pytest.raises(
        RuntimeError,
        match="closed",
    ):
        client.list_dir("content")

    http_session.request.assert_not_called()


def test_response_is_closed_when_status_check_fails(
    client,
    http_session,
):
    response = _response(status=500)
    response.raise_for_status.side_effect = requests.HTTPError("failed")
    http_session.request.return_value = response

    with pytest.raises(requests.HTTPError):
        client.list_dir("content")

    response.close.assert_called_once_with()
