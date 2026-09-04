import pytest
import requests

from data.crypto_quant.http import HttpRequestError, JsonHttpClient


class FakeResponse:
    def __init__(self, status_code, payload, *, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, url, *, params=None, timeout=None):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FalseyFakeSession(FakeSession):
    def __bool__(self):
        return False


def test_retries_429_using_retry_after():
    session = FakeSession([
        FakeResponse(429, {}, headers={"Retry-After": "2"}),
        FakeResponse(200, {"data": [1]}),
    ])
    sleeps = []
    client = JsonHttpClient(
        session, timeout=3, max_attempts=3, sleep=sleeps.append, random_fn=lambda: 0
    )
    assert client.get_json("https://example.test") == {"data": [1]}
    assert sleeps == [2.0]


def test_retries_500_then_succeeds():
    session = FakeSession([FakeResponse(500, {}), FakeResponse(200, {"ok": True})])
    client = JsonHttpClient(
        session, timeout=3, max_attempts=2, sleep=lambda _: None, random_fn=lambda: 0
    )
    assert client.get_json("https://example.test") == {"ok": True}


def test_does_not_retry_400():
    client = JsonHttpClient(FakeSession([FakeResponse(400, {"code": -1121, "msg": "Invalid symbol."})]), max_attempts=3)
    with pytest.raises(HttpRequestError, match="HTTP 400") as exc_info:
        client.get_json("https://example.test")
    assert exc_info.value.binance_code == -1121


def test_rejects_non_json_success():
    response = FakeResponse(200, ValueError("invalid json"))
    client = JsonHttpClient(FakeSession([response]), max_attempts=1)
    with pytest.raises(HttpRequestError, match="invalid JSON"):
        client.get_json("https://example.test")


def test_retries_connection_error_with_exponential_backoff():
    session = FakeSession([
        requests.ConnectionError("offline"),
        FakeResponse(200, {"ok": True}),
    ])
    sleeps = []
    client = JsonHttpClient(
        session, max_attempts=2, sleep=sleeps.append, random_fn=lambda: 0.25
    )

    assert client.get_json("https://example.test") == {"ok": True}
    assert sleeps == [1.25]


def test_retries_timeout_then_succeeds():
    session = FakeSession([
        requests.exceptions.Timeout("timed out"),
        FakeResponse(200, {"ok": True}),
    ])
    sleeps = []
    client = JsonHttpClient(
        session, max_attempts=2, sleep=sleeps.append, random_fn=lambda: 0
    )

    assert client.get_json("https://example.test") == {"ok": True}
    assert sleeps == [1.0]


def test_honors_zero_retry_after():
    session = FakeSession([
        FakeResponse(429, {}, headers={"Retry-After": "0"}),
        FakeResponse(200, {"ok": True}),
    ])
    sleeps = []
    client = JsonHttpClient(
        session, max_attempts=2, sleep=sleeps.append, random_fn=lambda: 0.25
    )

    assert client.get_json("https://example.test") == {"ok": True}
    assert sleeps == [0.0]


def test_does_not_retry_invalid_url():
    error = requests.exceptions.InvalidURL("invalid URL")
    session = FakeSession([error])
    sleeps = []
    client = JsonHttpClient(session, max_attempts=3, sleep=sleeps.append)

    with pytest.raises(
        HttpRequestError, match=r"HTTP request failed.*https://example\.test.*attempt 1/3"
    ) as exc_info:
        client.get_json("https://example.test")

    assert session.calls == 1
    assert sleeps == []
    assert exc_info.value.__cause__ is error


def test_uses_falsey_injected_session():
    session = FalseyFakeSession([FakeResponse(200, {"ok": True})])

    assert JsonHttpClient(session, max_attempts=1).get_json("https://example.test") == {
        "ok": True
    }
    assert session.calls == 1


def test_raises_after_final_server_error_attempt():
    session = FakeSession([FakeResponse(503, {}), FakeResponse(503, {})])
    sleeps = []
    client = JsonHttpClient(
        session, max_attempts=2, sleep=sleeps.append, random_fn=lambda: 0
    )

    with pytest.raises(HttpRequestError, match=r"HTTP 503.*attempt 2/2"):
        client.get_json("https://example.test")

    assert session.calls == 2
    assert sleeps == [1.0]
