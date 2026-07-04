"""Refiner — revises report based on critic feedback (3rd agent escalation)."""

from __future__ import annotations

import logging

from deep_reports.agents.base import AgentContext, AgentResult


logger = logging.getLogger("deep_reports.refiner")


SYSTEM_PROMPT = """You are a report refiner.

Your task: Given a report and critic feedback, improve the report to address the issues.
You MUST:
- Only modify sections that the critics flagged as problematic
- Preserve any good content — do not rewrite the entire report
- Address specific feedback points directly
- Do NOT add new content not supported by sources
- Keep the same structure (Executive Summary, Core Primitive, Key Patterns, etc.)

If there are no blockers from critics, make minor improvements only.

Output the revised report in full (Markdown format).
"""


class Refiner:
    """
    Agent that revises report based on critic feedback.

    Activates only on 3+ iteration plateau (tri-agent escalation).
    Operates on blockers only — does NOT add new content.
    """

    name = "refiner"
    role = "refiner"
    persona = SYSTEM_PROMPT

    def invoke(self, ctx: AgentContext) -> AgentResult:
        """Refine report based on critic feedback."""
        report_md = ctx.state.get("report_md", "")
        evidence_issues = ctx.state.get("evidence_issues", [])
        quality_scores = ctx.state.get("quality_scores", {})
        quality_summary = ctx.state.get("quality_summary", "")
        iteration = ctx.state.get("iteration", 1)

        feedback_parts = []

        if evidence_issues:
            issues_md = "\n".join(
                f"- [{i.get('severity', '?')}] {i.get('issue_type', '?')}: "
                f"{i.get('location', '?')} — {i.get('suggested_fix', 'fix needed')}"
                for i in evidence_issues
            )
            feedback_parts.append(f"## Evidence Issues\n{issues_md}\n")

        if quality_scores:
            scores_md = "\n".join(
                f"- {k}: {v}/10" for k, v in quality_scores.items()
            )
            feedback_parts.append(f"## Quality Scores\n{scores_md}\n")

        if quality_summary:
            feedback_parts.append(f"## Quality Critic Summary\n{quality_summary}\n")

        prompt = (
            f"## Current Report (iteration {iteration})\n\n{report_md}\n\n"
            + "\n\n".join(feedback_parts) +
            "\n\n## Task\nRevise the report to address the above feedback. "
            "Output the FULL revised report (not a diff)."
        )

        budget = ctx.get_budget()
        resp = ctx.provider.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.persona,
            temperature=0.0,
            max_tokens=4096,
            budget=budget,
        )

        return AgentResult(
            content=resp.content,
            structured=None,
            cost_usd=resp.cost_usd,
            tokens={
                "input": resp.input_tokens,
                "output": resp.output_tokens,
            },
            agent_name=self.name,
        )
