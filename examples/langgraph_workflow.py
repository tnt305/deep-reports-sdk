"""LangGraph workflow — same pipeline on LangGraph backend."""

from deep_reports import DeepReport

dr = DeepReport(
    sources=["./my-project/"],
    question="What are the main architectural patterns?",
    provider="anthropic",
    framework="langgraph",  # Switch backend
    max_cost_usd=1.0,
)

result = dr.generate()
print(result["report_md"])
