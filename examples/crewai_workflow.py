"""CrewAI workflow — same pipeline on CrewAI backend."""

from deep_reports import DeepReport

dr = DeepReport(
    sources=["./my-project/"],
    question="How does data flow through the system?",
    provider="anthropic",
    framework="crewai",  # Switch backend
    max_cost_usd=1.0,
)

result = dr.generate()
print(result["report_md"])
