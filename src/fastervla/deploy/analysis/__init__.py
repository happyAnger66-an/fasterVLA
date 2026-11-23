"""Offline analysis for robot-live traces (self-contained in fasterVLA)."""

from fastervla.deploy.analysis.metrics import wide_row
from fastervla.deploy.analysis.summarize import analyze_results_dir, analyze_run

__all__ = ["analyze_run", "analyze_results_dir", "wide_row"]
