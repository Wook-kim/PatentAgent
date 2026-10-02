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
