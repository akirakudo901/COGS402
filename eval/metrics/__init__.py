"""Evaluation helpers and metrics for the LLM–Prolog pipeline."""

from .well_defined_nl_symbol import (
    NlSymbolWellDefinedOutcome,
    WellDefinedFailureCategory,
    WellDefinedNlSymbolSummary,
    classify_swipl_pyswip_exception,
    initial_premises_from_nl_symbol_converter,
    log_nl_symbol_well_defined_tally,
    nl_symbol_conversion_assess,
    nl_symbol_conversion_is_well_defined,
    premises_for_nl_symbol_validity_check,
    selector_failed_under_well_defined_symbols,
    summarize_well_defined_nl_symbol_metrics,
    tally_nl_symbol_well_defined_outcomes,
)

__all__ = [
    "NlSymbolWellDefinedOutcome",
    "WellDefinedFailureCategory",
    "WellDefinedNlSymbolSummary",
    "classify_swipl_pyswip_exception",
    "initial_premises_from_nl_symbol_converter",
    "log_nl_symbol_well_defined_tally",
    "nl_symbol_conversion_assess",
    "premises_for_nl_symbol_validity_check",
    "nl_symbol_conversion_is_well_defined",
    "selector_failed_under_well_defined_symbols",
    "summarize_well_defined_nl_symbol_metrics",
    "tally_nl_symbol_well_defined_outcomes",
]
