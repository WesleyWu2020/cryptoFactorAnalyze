import unittest
from datetime import date
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pipeline_time_utils import add_months, previous_month_window  # noqa: E402


class PipelineTimeUtilsTest(unittest.TestCase):
    def test_add_months_rollover(self):
        self.assertEqual(add_months(date(2025, 12, 1), 1), date(2026, 1, 1))
        self.assertEqual(add_months(date(2025, 1, 1), 14), date(2026, 3, 1))

    def test_previous_month_window(self):
        start_, end_ = previous_month_window(date(2026, 1, 1))
        self.assertEqual(start_, date(2025, 12, 1))
        self.assertEqual(end_, date(2025, 12, 31))

        start_, end_ = previous_month_window(date(2024, 3, 1))
        self.assertEqual(start_, date(2024, 2, 1))
        self.assertEqual(end_, date(2024, 2, 29))


if __name__ == "__main__":
    unittest.main()
