"""Student-assignment code bundled with SFUSD Zone."""

import sys

from . import student_assignment

# Preserve the implementation's existing absolute imports within this package.
sys.modules.setdefault("student_assignment", student_assignment)
