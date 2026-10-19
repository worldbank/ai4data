"""End-to-end reference implementation of the AI for Data - Data for AI pattern.

The modules run in the order of the pipeline:

    records   load the series, the values, and the survey record
    validate  structure (JSON Schema), completeness (profile), validity (rules)
    publish   a catalog directory with records, values, and schema.org files,
              and optionally a NADA instance
    (server)  an MCP server over the catalog, run as a subprocess by run.py
    answer    an answer composed from the MCP tool results with claim tags
    verify    every claim tag checked with the PCN engine
    report    one Markdown and one JSON report of the run
"""

__all__ = ["answer", "publish", "records", "report", "validate", "verify"]
