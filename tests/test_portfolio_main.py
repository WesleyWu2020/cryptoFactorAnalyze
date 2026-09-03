import unittest
import pathlib
import sys
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestMainCLI(unittest.TestCase):
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "portfolio" / "main.py"), "--help"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--factors", result.stdout)

    def test_missing_required_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "portfolio" / "main.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
