# SPDX-License-Identifier: MIT
"""Tests for the filesystem-safe parts of the manual Colab wrapper."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

try:
    from .colab_manual import _normalise_zip_member, extract_bundle
except ImportError:  # pragma: no cover - supports direct execution
    from colab_manual import _normalise_zip_member, extract_bundle  # type: ignore


class ColabManualTests(unittest.TestCase):
    def test_windows_member_names_become_directories(self) -> None:
        self.assertEqual(
            _normalise_zip_member(r"trainer\colab_generate.py"),
            ("trainer", "colab_generate.py"),
        )

    def test_traversal_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _normalise_zip_member(r"..\outside.txt")

    def test_bundle_extracts_windows_style_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.zip"
            destination = root / "content" / "project"
            destination.parent.mkdir()
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(r"trainer\colab_generate.py", "print('ok')\n")
                zf.writestr(r"include\tetra\schema.hpp", "// schema\n")

            extract_bundle(archive, destination)
            self.assertEqual(
                (destination / "trainer" / "colab_generate.py").read_text(),
                "print('ok')\n",
            )
            self.assertEqual(
                (destination / "include" / "tetra" / "schema.hpp").read_text(),
                "// schema\n",
            )


if __name__ == "__main__":
    unittest.main()
