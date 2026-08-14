"""Walker inspection view — the debug surface, not the product.

The product is the JSON snapshot pushed to Finviz
(dilution/finviz_payload.py → scripts/push_finviz.py). The Flask dashboard
that used to render cards here is gone; what remains is `inspect.py`, which
dumps the raw ledger state behind those cards for debugging a walk.

Launched by run_inspect.py, bound to loopback. One route group:
  /inspect                                   ticker picker
  /inspect/<ticker>                          everything for one issuer
  /inspect/<ticker>/raw/<accession>[?doc=]   raw filing markdown
"""
