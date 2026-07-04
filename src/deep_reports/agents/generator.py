"""ReportGenerator — produces report content from sources + question."""

from __future__ import annotations

import logging
from pathlib import Path

from typing import Callable

from deep_reports.agents.base import AgentContext, AgentResult
from deep_reports.security import get_allowed_roots, validate_paths


logger = logging.getLogger("deep_reports.generator")


SYSTEM_PROMPT = """You are a technical research report writer.

Your task: Given a research question and source code, write a comprehensive, structured technical report.

Follow this structure:
1. **Executive Summary** — 2-3 sentence overview
2. **Core Primitive** — What is the fundamental building block?
3. **Key Patterns** — Major architectural patterns found
4. **Data Flow** — How data moves through the system
5. **Strengths & Limitations**
6. **Evidence** — Specific code citations with file:line references

Rules:
- Cite sources as: `file_path:line_number`
- Do NOT fabricate citations — only cite what you can verify in the sources
- Be precise and technical
- Use code snippets sparingly but meaningfully
"""


class ReportGenerator:
    """
    Agent that produces a structured technical report from source code + question.

    Responsibilities:
      - Read and summarize source files
      - Answer the research question with evidence from sources
      - Structure output as multi-section Markdown report
    """

    name = "generator"
    role = "generator"
    persona = SYSTEM_PROMPT

    def __init__(
        self,
        source_reader: Callable[[list[str]], dict[str, str]] | None = None,
        allowed_roots: list[str] | None = None,
    ):
        """
        Args:
            source_reader: callable that takes list[path] and returns {path: content}.
                          Defaults to reading files from disk.
            allowed_roots: list of allowed root paths for security validation.
                          Defaults to get_allowed_roots().
        """
        self._source_reader = source_reader or self._default_read_sources
        self._allowed_roots = allowed_roots or get_allowed_roots()

    def _default_read_sources(self, paths: list[str]) -> dict[str, str]:
        """Read files from disk with security validation at each I/O layer."""
        from pydantic import ValidationError

        result: dict[str, str] = {}
        try:
            validated = validate_paths(paths, allowed_roots=self._allowed_roots)
        except ValidationError as e:
            logger.warning(f"Source validation failed: {e}")
            return result

        for path in validated:
            p = Path(path)
            if p.is_file():
                try:
                    result[str(p)] = p.read_text(encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Could not read {p}: {e}")
            elif p.is_dir():
                for f in p.rglob("*.py"):
                    try:
                        result[str(f)] = f.read_text(encoding="utf-8")
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Could not read {f}: {e}")
        return result

    def _build_prompt(self, ctx: AgentContext) -> str:
        """Build the generation prompt from context."""
        question = ctx.state.get("question", "")
        sources_raw = ctx.state.get("sources", [])
        sources_content = self._source_reader(sources_raw)

        files_md = []
        for path, content in sources_content.items():
            files_md.append(f"--- {path} ---\n{content}")

        return (
            f"## Research Question\n{question}\n\n"
            f"## Source Files\n\n" + "\n\n".join(files_md)
        )

    def invoke(self, ctx: AgentContext) -> AgentResult:
        """Generate report by calling the provider with sources + question."""
        prompt = self._build_prompt(ctx)
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
