from __future__ import annotations


class LangChainJudgeClient:
    """Adapts LangChain ChatOpenAI to EvalTrace's JudgeClient protocol."""

    def __init__(self, llm):
        self.llm = llm

    def complete(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content
