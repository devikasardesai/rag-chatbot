"""Shared pytest fixtures for the RAG FAQ assistant test suite."""
from __future__ import annotations

import pytest

from src import config

SAMPLE_MARKDOWN = """# HDFC Small Cap Fund - Direct Growth

HDFC Small Cap Fund is an equity oriented fund that seeks to invest in small cap stocks.

## Minimum investment

| Mode | Minimum amount |
| --- | --- |
| Lump sum | Rs. 10,000 |
| SIP | Rs. 500 per month |

## Exit load

| Holding period | Exit load |
| --- | --- |
| 0-12 months | 1.0% |
| 12-24 months | 0.75% |
| Above 24 months | Nil |

## Fees and expenses

| Particulars | % of NAV |
| --- | --- |
| Management fee | 0.50% |
| Total expense ratio | 1.19% |

## Benchmark and risk

The benchmark of the scheme is Nifty Smallcap 250 TRI. The riskometer category is Very High.

## Tax and documents

To download the capital-gains statement, use the investor portal and select the statement
type and the financial year.
"""


@pytest.fixture()
def corpus() -> list[config.SourceSpec]:
    """The five allowlisted corpus entries from config."""
    return list(config.CORPUS)


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    """Point config data and log directories at a temporary directory."""
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", data_dir / "processed")
    monkeypatch.setattr(config, "CHROMA_DIR", data_dir / "chroma")
    monkeypatch.setattr(config, "LOG_DIR", log_dir)
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "corpus_manifest.json")
    monkeypatch.setattr(
        config, "CHUNKING_REPORT_PATH", log_dir / "chunking_report.json"
    )
    return tmp_path


@pytest.fixture()
def sample_markdown() -> str:
    """A realistic scheme-page Markdown fixture with headings, a fee table, an exit-load table."""
    return SAMPLE_MARKDOWN


@pytest.fixture()
def fake_specs() -> list[config.SourceSpec]:
    """Two non-corpus specs for allowlist tests; URLs are never fetched."""
    return [
        config.SourceSpec(
            url="https://example.com/mutual-funds/fake-large-cap",
            scheme_name="Fake Large Cap Fund",
            category="Large Cap",
            plan="Direct Growth",
        ),
        config.SourceSpec(
            url="https://example.com/mutual-funds/fake-elss",
            scheme_name="Fake ELSS Fund",
            category="ELSS",
            plan="Direct Growth",
        ),
    ]
