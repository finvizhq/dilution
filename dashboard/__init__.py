"""Flask dashboard for the dilution pipeline.

Reads directly from dilution.db — no API layer, no caching. Routes:
  /                  — list of tracked tickers
  /t/<ticker>        — single-ticker dilution detail page
"""
