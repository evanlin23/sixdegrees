from types import SimpleNamespace as NS

import pytest

anthropic = pytest.importorskip("anthropic")

from sixdegrees.planners import ClaudePlanner, PlannerError, echoable  # noqa: E402

GOOD = {"chain": [{"from": "Ann Lee", "to": "Bob Ray", "occasion": "Gala", "year": "2001", "query": "q1"},
                  {"from": "Bob Ray", "to": "Cat Moe", "occasion": "Summit", "year": "2005", "query": "q2"}],
        "notes": "via Bob"}


def text(t):
    return NS(type="text", text=t)


def submit(data, id_="toolu_1"):
    return NS(type="tool_use", name="submit_chain", input=data, id=id_)


def msg(stop_reason, *content, iterations=None, stop_details=None):
    return NS(stop_reason=stop_reason, content=list(content), model="claude-opus-5-5",
              usage=NS(iterations=iterations or []), stop_details=stop_details)


class FakeStream:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    """Stands in for anthropic.Anthropic(); records every request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return FakeStream(self.replies.pop(0))


def planner(replies):
    client = FakeClient(replies)
    return ClaudePlanner(client=client), client


def test_plan_uses_web_search_strict_tool_and_fallbacks():
    p, client = planner([msg("tool_use", text("Found it."), submit(GOOD))])
    links = p.plan("Ann Lee", "Cat Moe")
    assert [(l.a, l.b, l.occasion) for l in links] == [("Ann Lee", "Bob Ray", "Gala"), ("Bob Ray", "Cat Moe", "Summit")]
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["output_config"] == {"effort": "high"}
    assert "thinking" not in req  # always on for Opus 5.5; effort is the control
    names = [t["name"] for t in req["tools"]]
    assert names == ["web_search", "submit_chain"]
    assert req["tools"][1]["strict"] is True


def test_pause_turn_is_resumed_without_an_extra_user_message():
    paused = msg("pause_turn", NS(type="server_tool_use", id="srv_1"))
    p, client = planner([paused, msg("tool_use", submit(GOOD))])
    assert p.plan("Ann Lee", "Cat Moe")
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    assert second[1]["content"] == paused.content


def test_invalid_chain_goes_back_as_tool_error_then_succeeds():
    bad = {"chain": [{"from": "Ann Lee", "to": "Dan Fox", "occasion": "", "year": "", "query": ""}], "notes": ""}
    p, client = planner([msg("tool_use", submit(bad, "toolu_bad")), msg("tool_use", submit(GOOD, "toolu_2"))])
    assert [l.b for l in p.plan("Ann Lee", "Cat Moe")] == ["Bob Ray", "Cat Moe"]
    reply = client.requests[1]["messages"][-1]
    assert reply["role"] == "user"
    result = reply["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_bad" and result["is_error"]


def test_text_json_is_accepted_and_missing_tool_call_is_nudged():
    p, client = planner([msg("end_turn", text("I think it's Bob.")), msg("end_turn", text('{"chain": [{"from": "Ann Lee",'
                         ' "to": "Cat Moe"}], "notes": ""}'))])
    assert [l.b for l in p.plan("Ann Lee", "Cat Moe")] == ["Cat Moe"]
    assert "submit_chain" in client.requests[1]["messages"][-1]["content"]


def test_refusal_and_max_tokens_give_up_cleanly():
    p, _ = planner([msg("refusal", stop_details=NS(category="cyber"))])
    with pytest.raises(PlannerError):
        p.plan("Ann Lee", "Cat Moe")
    p, _ = planner([msg("max_tokens", text("..."))])
    assert p.repair(["Ann Lee"], "Cat Moe", [], []) is None


def test_answers_are_cached_between_runs():
    p, _ = planner([msg("tool_use", submit(GOOD))])
    p.plan("Ann Lee", "Cat Moe")
    again, client = planner([])
    assert [l.b for l in again.plan("Ann Lee", "Cat Moe")] == ["Bob Ray", "Cat Moe"]
    assert client.requests == []


def _api_error(cls, status, message):
    import anthropic._base_client as base

    httpx = getattr(base, "httpx2", None) or base.httpx
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls(message, response=response, body={"error": {"message": message}})


def test_rejected_optional_features_are_dropped_and_retried():
    err = _api_error(anthropic.BadRequestError, 400, "web_search is not enabled for this organization")
    p, client = planner([err, msg("tool_use", submit(GOOD))])
    assert p.plan("Ann Lee", "Cat Moe")
    assert [t["name"] for t in client.requests[1]["tools"]] == ["submit_chain"]


def test_auth_errors_become_planner_errors():
    p, _ = planner([_api_error(anthropic.AuthenticationError, 401, "invalid x-api-key")])
    with pytest.raises(PlannerError, match="ANTHROPIC_API_KEY"):
        p.plan("Ann Lee", "Cat Moe")
    p, _ = planner([_api_error(anthropic.BadRequestError, 400, "something else is wrong")])
    with pytest.raises(PlannerError, match="something else"):
        p.plan("Ann Lee", "Cat Moe")


def test_echoable_drops_unreplayable_blocks_before_a_fallback():
    blocks = [NS(type="thinking"), text("partial"), NS(type="server_tool_use", id="a"),
              NS(type="web_search_tool_result", tool_use_id="a"), NS(type="server_tool_use", id="orphan"),
              NS(type="tool_use", id="t"), NS(type="fallback"), NS(type="thinking"), text("rest")]
    kept = [b.type for b in echoable(blocks)]
    assert kept == ["text", "server_tool_use", "web_search_tool_result", "fallback", "thinking", "text"]
    assert echoable(blocks[:3]) == blocks[:3]  # no fallback: unchanged


def _sse(events):
    import json as _json

    return "".join(f"event: {e['type']}\ndata: {_json.dumps(e)}\n\n" for e in events).encode()


def test_real_sdk_request_and_stream_parsing():
    """Run the real SDK against a mock transport: checks the wire format and stream accumulation."""
    import json as _json

    import anthropic._base_client as base

    httpx = getattr(base, "httpx2", None) or base.httpx
    seen = {}
    tool_json = _json.dumps(GOOD)

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = _json.loads(request.content)
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                  "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                                                  "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "submit_chain", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": tool_json[:40]}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": tool_json[40:]}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
             "usage": {"output_tokens": 50}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    client = anthropic.Anthropic(api_key="test-key", base_url="http://mock.local", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler)))
    links = ClaudePlanner(client=client).plan("Ann Lee", "Cat Moe")
    assert [l.b for l in links] == ["Bob Ray", "Cat Moe"]
    body, headers = seen["body"], seen["headers"]
    assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert body["fallbacks"] == "default" and body["stream"] is True and body["model"] == "claude-opus-5-5"
    assert body["tools"][0] == {"type": "web_search_20260209", "name": "web_search", "max_uses": 10}
    assert body["tools"][1]["strict"] is True and body["tools"][1]["eager_input_streaming"] is True
    assert body["output_config"] == {"effort": "high"}
