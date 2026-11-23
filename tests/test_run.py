"""Tests for the launcher without a GPU, network, or external libraries."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_catalog_contains_granite(self):
        models = run.load_catalog()
        granite = models["granite"]
        self.assertEqual(granite["type"], "text")
        self.assertEqual(len(granite["sha256"]), 64)
        self.assertTrue(granite["url"].startswith("https://"))

    def test_platform_suffixes(self):
        scenarios = [
            ("Linux", "x86_64", ("linux-x64", "-bin-ubuntu-x64.tar.gz")),
            ("Linux", "aarch64", ("linux-arm64", "-bin-ubuntu-arm64.tar.gz")),
            ("Darwin", "arm64", ("macos-arm64", "-bin-macos-arm64.tar.gz")),
            ("Darwin", "x86_64", ("macos-x64", "-bin-macos-x64.tar.gz")),
            ("Windows", "AMD64", ("windows-x64", "-bin-win-cpu-x64.zip")),
            ("Windows", "ARM64", ("windows-arm64", "-bin-win-cpu-arm64.zip")),
        ]
        for system, machine, expected in scenarios:
            with self.subTest(system=system, machine=machine), \
                 patch("run.platform.system", return_value=system), \
                 patch("run.platform.machine", return_value=machine):
                self.assertEqual(run.platform_archive_suffix(), expected)

    def test_asset_selection_rejects_cuda(self):
        asset = {
            "name": "llama-b42-bin-ubuntu-x64.tar.gz",
            "digest": "sha256:" + "a" * 64,
            "browser_download_url": "https://github.com/example/llama.tar.gz",
        }
        cuda = dict(asset, name="cudart-llama-b42-bin-ubuntu-x64.tar.gz")
        url, sha = run.select_asset([{"assets": [cuda, asset]}], "-bin-ubuntu-x64.tar.gz")
        self.assertEqual(url, asset["browser_download_url"])
        self.assertEqual(sha, "a" * 64)

    def test_offline_model_uses_verified_file(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            model = {
                "filename": "tiny.gguf",
                "sha256": hashlib.sha256(b"fake-model").hexdigest(),
                "url": "https://example.org/tiny.gguf",
            }
            path = cache / "models" / "tiny.gguf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fake-model")
            with patch.object(run, "CACHE", cache), \
                 patch("run.urllib.request.urlopen", side_effect=AssertionError("network used")):
                self.assertEqual(run.ensure_model(model, offline=True), path)

    def test_offline_model_refuses_corrupted_file(self):
        with tempfile.TemporaryDirectory() as temp:
            model = {
                "filename": "tiny.gguf",
                "sha256": "0" * 64,
                "url": "https://example.org/tiny.gguf",
            }
            with patch.object(run, "CACHE", Path(temp)):
                with self.assertRaisesRegex(RuntimeError, "without --offline"):
                    run.ensure_model(model, offline=True)

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("../../unsafe.txt", "malicious")
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                run.unpack(archive, Path(temp) / "extract")
            self.assertFalse((Path(temp) / "unsafe.txt").exists())

    def test_tar_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                content = b"malicious"
                member = tarfile.TarInfo("../../unsafe.txt")
                member.size = len(content)
                package.addfile(member, io.BytesIO(content))
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                run.unpack(archive, Path(temp) / "extract")

    def test_command_is_cpu_only(self):
        model = {"name": "Granite test"}
        with patch("run.load_catalog", return_value={"granite": model}), \
             patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
             patch("run.ensure_model", return_value=Path("/tmp/granite.gguf")), \
             patch("run.subprocess.call", return_value=0) as call, \
             patch.object(sys, "argv", ["run.py", "Hello", "--threads", "2"]):
            self.assertEqual(run.main(), 0)
        command = call.call_args.args[0]
        self.assertIn("--gpu-layers", command)
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
        self.assertIn("--single-turn", command)
        self.assertEqual(command[command.index("--prompt") + 1], "Hello")

    def test_no_network_for_list(self):
        with patch("run.ensure_runtime", side_effect=AssertionError("network attempted")), \
             patch.object(sys, "argv", ["run.py", "--list"]):
            self.assertEqual(run.main(), 0)


if __name__ == "__main__":
    unittest.main()
