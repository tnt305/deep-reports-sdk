"""QualityCritic — 8-dimension rubric scoring (I9 spec)."""

from __future__ import annotations

import json
import logging
import re

from deep_reports.agents.base import AgentContext, AgentResult


logger = logging.getLogger("deep_reports.critic_quality")


SYSTEM_PROMPT = """You are a quality critic reviewing a technical research report.

Score the report on these 8 dimensions (each 1-10):

1. **Depth** — Does the report deeply analyze the topic, or only scratch the surface?
2. **Breadth** — Does it cover all relevant aspects of the codebase?
3. **Evidence** — Are claims well-supported by source code citations?
4. **Flow** — Is the report well-organized and easy to follow?
5. **Viz** — Are diagrams or visual elements used effectively?
6. **Comparability** — Does it relate to known patterns (e.g. design patterns, frameworks)?
7. **Actionability** — Can a reader act on the findings?
8. **Clarity** — Is the writing clear, precise, and free of jargon?

Respond with JSON only (no markdown, no preamble):

{
  "scores": {
    "depth": 7,
    "breadth": 6,
    "evidence": 8,
    "flow": 7,
    "viz": 4,
    "comparability": 5,
    "actionability": 6,
    "clarity": 8
  },
  "overall": 6.5,
  "summary": "1-2 sentence assessment",
  "recommendations": ["recommendation 1", "recommendation 2"]
}

Rules:
- Score honestly. A 5/10 means average; don't inflate scores.
- If a section is missing, score it 1-3.
- Evidence dimension is the most important — penalize heavily if claims are unsourced.
"""


class QualityCritic:
    """
    Agent that scores a report on an 8-dimension rubric.

    I9 spec: depth, breadth, evidence, flow, viz, comparability, actionability, clarity.
    Outputs structured JSON with per-dimension scores + overall.
    """

    name = "quality_critic"
    role = "quality_critic"
    persona = SYSTEM_PROMPT

    def _parse_json(self, content: str) -> dict | None:
        """Extract and parse JSON from model response using balanced-brace extraction."""
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            start = content.find("{")
            if start == -1:
                return None
            depth = 0
            end = -1
            for i, ch in enumerate(content[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    return None
            return None

    def invoke(self, ctx: AgentContext) -> AgentResult:
        """Score the report on 8 dimensions."""
        report_md = ctx.state.get("report_md", "")

        prompt = f"## Report to Score\n\n{report_md}"

        budget = ctx.get_budget()
        resp = ctx.provider.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.persona,
            temperature=0.0,
            max_tokens=1024,
            budget=budget,
        )

        parsed = self._parse_json(resp.content)
        overall_score = None

        if parsed and "overall" in parsed:
            overall_score = float(parsed["overall"])
            logger.info(f"QualityCritic: overall score = {overall_score}/10")
        elif parsed:
            scores = parsed.get("scores", {})
            if scores:
                vals = [v for v in scores.values() if isinstance(v, (int, float))]
                overall_score = sum(vals) / len(vals) if vals else None

        return AgentResult(
            content=resp.content,
            structured=parsed,
            cost_usd=resp.cost_usd,
            tokens={
                "input": resp.input_tokens,
                "output": resp.output_tokens,
            },
            agent_name=self.name,
        )
