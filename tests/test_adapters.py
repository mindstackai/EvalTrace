from judge.adapters import LangChainJudgeClient


class FakeLLM:
    """Mimics LangChain ChatOpenAI interface."""
    class Response:
        def __init__(self, content):
            self.content = content

    def invoke(self, prompt):
        return self.Response(f"echo: {prompt}")


def test_langchain_adapter_calls_invoke():
    llm = FakeLLM()
    client = LangChainJudgeClient(llm)
    result = client.complete("hello")
    assert result == "echo: hello"


def test_langchain_adapter_satisfies_protocol():
    """Verify it has the .complete(str) -> str method signature."""
    client = LangChainJudgeClient(FakeLLM())
    assert hasattr(client, "complete")
    assert callable(client.complete)
