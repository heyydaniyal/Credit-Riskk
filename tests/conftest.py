"""Shared test setup.

The API logs every /score call. Tests must never write into the service's
real request log (it feeds the monitoring tab), so the whole session points
the log at a scratch file before any app module is imported.
"""

import os
import tempfile

os.environ["CREDIT_RISK_REQUEST_LOG"] = os.path.join(
    tempfile.mkdtemp(prefix="credit-risk-tests-"), "request_log.sqlite")
