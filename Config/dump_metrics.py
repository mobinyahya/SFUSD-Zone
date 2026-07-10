"""Dump ALL_METRICS to CSV on stdout using dataclass reflection."""

import csv
import sys
from dataclasses import fields

from Config.metrics_config import ALL_METRICS, MetricSpec

writer = csv.writer(sys.stdout)
header = [f.name for f in fields(MetricSpec)]
writer.writerow(header)
for m in ALL_METRICS:
    writer.writerow([getattr(m, name) for name in header])
