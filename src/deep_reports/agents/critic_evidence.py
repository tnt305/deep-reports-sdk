"""EvidenceCritic — evaluates claim-to-source alignment (I7 spec)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from deep_reports.agents.base import AgentContext, AgentResult
from deep_reports.security import get_allowed_roots, validate_paths


logger = logging.getLogger("deep_reports.critic_evidence")


# Strict JSON schema enforced on the critic's structured output
EVIDENCE_ISSUE_SCHEMA = {
    "type": "object",
    "required": ["severity", "issue_type", "location", "suggested_fix"],
    "properties": {
        "severity": {
            "enum": ["blocker", "major", "minor"],
            "description": "blocker = must fix before report is valid; major = significant quality issue; minor = optional improvement",
        },
        "issue_type": {
            "enum": [
                "fabrication",      # cited content doesn't exist in source
                "weak_evidence",    # citation exists but doesn't support the claim
                "contradicts_source",  # cited source contradicts the claim
                "no_citation",     # claim has no source citation
                "wrong_citation",  # citation format is wrong (file not found, line out of range)
                "level_mismatch",  # generalization at wrong abstraction level
            ],
        },
        "location": {
            "type": "string",
            "description": "Where in the report: section heading or file:line reference",
        },
        "suggested_fix": {
            "type": "string",
            "description": "Concrete fix — what to change to resolve the issue",
        },
    },
}

ISSUES_SCHEMA = {
    "type": "object",
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": EVIDENCE_ISSUE_SCHEMA,
        },
        "summary": {
            "type": "string",
            "description": "1-2 sentence overall assessment",
        },
    },
}


SYSTEM_PROMPT = """You are an evidence critic reviewing a technical research report.

Your job: Verify every claim has supporting evidence in the source files.
For each issue found, output JSON with this exact structure:

{
  "issues": [
    {
      "severity": "blocker | major | minor",
      "issue_type": "fabrication | weak_evidence | contradicts_source | no_citation | wrong_citation | level_mismatch",
      "location": "section name or file:line",
      "suggested_fix": "what to change"
    }
  ],
  "summary": "overall assessment"
}

Severity rules:
- BLOCKER: fabricated content (citation to non-existent text), or cited file:line out of range
- MAJOR: claim contradicts source, or citation doesn't support claim
- MINOR: missing citation for minor claim, or weak/indirect evidence

Check EVERY citation by reading the actual source file. Do NOT assume a citation is valid.
Report line numbers as written in the report citation, not relative to file.
"""


class EvidenceCritic:
    """
    Agent that evaluates claim-to-source alignment.

    I7 spec: Every citation must be verifiable by reading the actual source file.
    Uses jsonschema to enforce EVIDENCE_ISSUE_SCHEMA on structured output.
    """

    name = "evidence_critic"
    role = "evidence_critic"
    persona = SYSTEM_PROMPT

    def __init__(self, allowed_roots: list[str] | None = None):
        """
        Args:
            allowed_roots: list of allowed root paths for security validation.
                          Defaults to get_allowed_roots().
        """
        self._allowed_roots = allowed_roots or get_allowed_roots()

    def _read_source_for_verification(self, sources: list[str]) -> dict[str, str]:
        """Read source files so the critic can verify citations, with security validation."""
        from pathlib import Path
        from pydantic import ValidationError

        result = {}
        try:
            validated = validate_paths(sources, allowed_roots=self._allowed_roots)
        except ValidationError as e:
            logger.warning(f"Source validation failed: {e}")
            return result

        for path in validated:
            p = Path(path)
            if p.is_file():
                try:
                    result[str(p)] = p.read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            elif p.is_dir():
                for f in p.rglob("*.py"):
                    try:
                        result[str(f)] = f.read_text(encoding="utf-8")
                    except Exception:  # noqa: BLE001
                        pass
        return result

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        """Extract and parse JSON from model response using balanced-brace extraction."""
        # Try markdown code block first (most reliable)
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Extract using balanced-brace counting (handles trailing "}" in text)
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
        """Evaluate report against sources."""
        report_md = ctx.state.get("report_md", "")
        sources = ctx.state.get("sources", [])
        sources_content = self._read_source_for_verification(sources)

        # Build context for the critic
        sources_md = "\n\n".join(
            f"--- {path} ---\n{content}"
            for path, content in sources_content.items()
        )

        prompt = (
            f"## Report to Review\n\n{report_md}\n\n"
            f"## Source Files (verify citations against these)\n\n{sources_md}"
        )

        budget = ctx.get_budget()
        resp = ctx.provider.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.persona,
            temperature=0.0,
            max_tokens=2048,
            budget=budget,
        )

        # Parse structured output
        parsed = self._parse_json(resp.content)
        structured = None
        blocker_count = 0

        if parsed:
            try:
                from jsonschema import validate
                validate(instance=parsed, schema=ISSUES_SCHEMA)
                structured = parsed
                blocker_count = sum(
                    1 for i in parsed.get("issues", [])
                    if i.get("severity") == "blocker"
                )
                logger.info(
                    f"EvidenceCritic: {len(parsed.get('issues', []))} issues "
                    f"({blocker_count} blockers)"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"JSON schema validation failed: {e}; using raw output")
                structured = {"raw": resp.content}

        return AgentResult(
            content=resp.content,
            structured=structured,
            cost_usd=resp.cost_usd,
            tokens={
                "input": resp.input_tokens,
                "output": resp.output_tokens,
            },
            agent_name=self.name,
        )
