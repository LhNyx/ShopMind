"""提取器乱码防线：tool-call 流式聚合切碎汉字时换 JSON 路径重试，仍中招则剥离。"""

from langchain_core.messages import AIMessage, HumanMessage

from shopmind.llm.extractor import LLMRequirementsExtractor, _has_mojibake, _strip_mojibake
from shopmind.schemas import Requirements


class _FakeStructured:
    def __init__(self, result) -> None:
        self._result = result

    async def ainvoke(self, prompt):
        return self._result


class _FakeModel:
    def __init__(self, structured_result, raw_json: str) -> None:
        self._structured = _FakeStructured(structured_result)
        self._raw_json = raw_json

    def with_structured_output(self, schema):
        return self._structured

    async def ainvoke(self, prompt):
        return AIMessage(content=self._raw_json)


async def test_mojibake_in_structured_output_retries_with_json_path():
    corrupted = Requirements(category="无线耳机", budget_max=1500, attributes=["主�降噪"])
    model = _FakeModel(
        corrupted,
        '{"category": "无线耳机", "budget_max": 1500, "attributes": ["主动降噪"]}',
    )
    req = await LLMRequirementsExtractor(model).extract(
        [HumanMessage(content="预算1500的主动降噪耳机")], None
    )
    assert req.attributes == ["主动降噪"]
    assert req.budget_max == 1500


async def test_mojibake_stripped_when_retry_also_corrupt():
    corrupted = Requirements(attributes=["主�降噪"])
    model = _FakeModel(corrupted, '{"attributes": ["主�降噪"]}')
    req = await LLMRequirementsExtractor(model).extract([HumanMessage(content="x")], None)
    assert not _has_mojibake(req)
    assert req.attributes == ["主降噪"]


def test_strip_mojibake_drops_fields_that_become_empty():
    req = _strip_mojibake(Requirements(category="�", attributes=["�", "防水"]))
    assert req.category is None
    assert req.attributes == ["防水"]
