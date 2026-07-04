"""Basic usage — generate a report from a local repository."""

from deep_reports import DeepReport

dr = DeepReport(
    sources=["./tests/fixtures/sample_repo/"],
    question="What is this library's core primitive?",
    provider="anthropic",
    framework="native",
    max_cost_usd=0.50,
)

result = dr.generate()
print(result["report_md"])
print(f"\nTotal cost: ${result['cost_usd']:.4f}")
