"""CLI entry point — Click commands for deep-reports."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from deep_reports._version import __version__
from deep_reports.config import DeepReportsConfig
from deep_reports.security import get_allowed_roots, validate_paths


@click.group()
@click.version_option(version=__version__)
def cli():
    """Deep Reports — Research report generator via LLM agents."""
    pass


@cli.command()
@click.option("--provider", default=None, help="LLM provider (anthropic, openai, litellm)")
@click.option("--framework", default=None, help="Orchestrator (native, langgraph, crewai)")
@click.option("--model", default=None, help="Model override")
@click.option(
    "--source", "sources", multiple=True, required=True,
    help="Source path (file or directory)"
)
@click.option("--question", required=True, help="Research question")
@click.option("--output", default=None, help="Output directory")
@click.option("--max-cost", default=None, type=float, help="Budget cap in USD")
@click.option("--agents", default=None, help="Agent set (all, core, minimal)")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def generate(
    provider, framework, model, sources, question, output, max_cost, agents, verbose
):
    """Generate a research report from source code."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build config with overrides
    cfg = DeepReportsConfig()
    if provider:
        cfg = cfg.with_overrides(provider=provider)
    if framework:
        cfg = cfg.with_overrides(framework=framework)
    if model:
        cfg = cfg.with_overrides(model=model)
    if output:
        cfg = cfg.with_overrides(output_dir=output)
    if max_cost is not None:
        cfg = cfg.with_overrides(max_cost_usd=max_cost)
    if agents:
        cfg = cfg.with_overrides(agents=agents)

    # Validate source paths at CLI boundary
    try:
        roots = get_allowed_roots()
        validated = validate_paths(list(sources), allowed_roots=roots)
        click.echo(f"Validated {len(validated)} source path(s)", err=True)
    except Exception as e:
        raise click.ClickException(str(e))

    # Run pipeline
    from deep_reports import DeepReport

    try:
        dr = DeepReport(
            sources=list(sources),
            question=question,
            config=cfg,
        )
        final = dr.generate()

        report_md = final.get("report_md", "")
        click.echo(f"\nReport written: {len(report_md)} chars", err=True)
        click.echo(f"Total cost: ${final.get('cost_usd', 0):.4f}", err=True)

        output_path = Path(cfg.output_dir) / "report.md"
        click.echo(f"Output: {output_path}")

    except Exception as e:
        raise click.ClickException(f"Pipeline failed: {e}")


@cli.command()
def version():
    """Show version and exit."""
    from deep_reports import __version__
    click.echo(f"deep-reports {__version__}")


if __name__ == "__main__":
    cli()
