import json

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from patentagent.config import Settings
from patentagent.llm import VisionClient
from patentagent.schemas import Identity


@pytest.mark.parametrize("content,finish,valid", [
    ('{"compound_id":"7","evidence_text":"Compound 7"}', "stop", True),
    ('```json\n{"compound_id":null,"evidence_text":null}\n```', "stop", True),
    ('{"compound_id":"7","surprise":true}', "stop", False),
    ('{"compound_id":"7"}', "length", False),
    ("not json", "stop", False),
])
def test_openai_compatible_contract(tmp_path, content, finish, valid):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={
            "id": "local-test", "object": "chat.completion", "created": 0, "model": "vision",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
        })
    client = VisionClient(Settings(llm_model="vision", llm_base_url="http://localhost:9000/v1"))
    client.client.close()
    client.client = OpenAI(
        api_key="test-key", base_url="http://localhost:9000/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        if valid:
            assert isinstance(client.extract("read ID", Identity, [image]), Identity)
        else:
            with pytest.raises(ValueError):
                client.extract("read ID", Identity, [image])
        assert str(captured[0].url) == "http://localhost:9000/v1/chat/completions"
        body = json.loads(captured[0].content)
        assert body["model"] == "vision"
        assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        client.client.close()


@pytest.mark.parametrize("mode", ["json_schema", "none"])
def test_response_format_setting(tmp_path, mode):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "local-test", "object": "chat.completion", "created": 0, "model": "vision",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": '{"compound_id":"7","evidence_text":"Compound 7"}'}}],
        })
    client = VisionClient(Settings(llm_model="vision", llm_response_format=mode))
    client.client.close()
    client.client = OpenAI(
        api_key="test-key", base_url="http://localhost:9000/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        client.extract("read ID", Identity, [image])
        if mode == "json_schema":
            fmt = captured[0]["response_format"]
            assert fmt["type"] == "json_schema"
            assert fmt["json_schema"]["name"] == "Identity"
        else:
            assert "response_format" not in captured[0]
    finally:
        client.client.close()


def test_invalid_response_format():
    with pytest.raises(ValueError):
        Settings(llm_model="vision", llm_response_format="json_object").require_llm()
