"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def row(report: str, name: str) -> tuple[float, float]:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


def test_report_quotes_table_ii_correctly(report, targets):
    assert row(report, "Paper (Table II, 100%)") == (targets["market1501"]["map"], targets["market1501"]["rank1"])


def test_market_map_matches_table_ii(report, targets):
    assert row(report, "**Ours**")[0] == pytest.approx(targets["market1501"]["map"], abs=targets["tolerance_pp"])


def test_market_rank1_matches_table_ii(report, targets):
    assert row(report, "**Ours**")[1] == pytest.approx(targets["market1501"]["rank1"], abs=targets["tolerance_pp"])
