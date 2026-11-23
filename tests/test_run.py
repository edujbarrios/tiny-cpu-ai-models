"""Tests for the launcher without a GPU, network, or external libraries."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_catalog_contains_granite(self):
        models = run.load_catalog()
        granite = models["granite-350m"]
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

    def test_cpu_runtime_manifest_is_pinned_and_checksum_verified(self):
        runtime = run.load_runtimes()["llama.cpp"]
        self.assertTrue(runtime["version"].startswith("b"))
        self.assertEqual(len(runtime["platforms"]), 6)
        for system, artifact in runtime["platforms"].items():
            with self.subTest(system=system):
                self.assertTrue(artifact["url"].startswith("https://github.com/"))
                self.assertTrue(artifact["filename"].startswith("llama-"))
                self.assertNotIn("cuda", artifact["filename"])
                self.assertEqual(len(artifact["sha256"]), 64)

    def test_runtime_offline_requires_downloaded_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(run, "CACHE", Path(temp)), \
                 patch("run.urllib.request.urlopen", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(RuntimeError, "without --offline"):
                    run.ensure_runtime(offline=True)

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
        model = {"name": "Granite test", "type": "text", "parameters": "350M", "aliases": ["granite"]}
        with patch("run.load_catalog", return_value={"granite-350m": model}), \
             patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
             patch("run.ensure_model", return_value=Path("/tmp/granite.gguf")), \
             patch("run.subprocess.call", return_value=0) as call, \
             patch.object(sys, "argv", ["run.py", "granite-350m", "Hello", "--threads", "2"]):
            self.assertEqual(run.main(), 0)
        command = call.call_args.args[0]
        self.assertIn("--gpu-layers", command)
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
        self.assertIn("--show-timings", command)
        self.assertNotIn("--no-show-timings", command)
        self.assertIn("--single-turn", command)
        self.assertEqual(command[command.index("--prompt") + 1], "Hello")


    def test_canonical_and_alias_model_ids(self):
        models = run.load_catalog()
        self.assertEqual(
            run.resolve_model(models, "granite-350m")[0], "granite-350m"
        )
        self.assertEqual(
            run.resolve_model(models, "granite")[0], "granite-350m"
        )


    def test_cpu_response_time_summary(self):
        output = io.StringIO()
        with patch("run.time.perf_counter", side_effect=[100.0, 103.25]), \
             patch("run.subprocess.call", return_value=0) as execute, \
             redirect_stderr(output):
            result = run.run_inference(["llama-cli"], Path("."), interactive=False)
        self.assertEqual(result, 0)
        execute.assert_called_once()
        self.assertIn("Response time: 3.25 s", output.getvalue())
        self.assertIn("model loading, prompt processing, and generation", output.getvalue())
        self.assertIn("tokens/s", output.getvalue())

    def test_interactive_reports_session_not_response_duration(self):
        output = io.StringIO()
        with patch("run.time.perf_counter", side_effect=[200.0, 202.4]), \
             patch("run.subprocess.call", return_value=0), \
             redirect_stderr(output):
            result = run.run_inference(["llama-cli"], Path("."), interactive=True)
        self.assertEqual(result, 0)
        self.assertIn("Chat session: 2.40 s", output.getvalue())
        self.assertIn("Per-response generation throughput", output.getvalue())
        self.assertNotIn("Response time:", output.getvalue())

    def test_failed_inference_does_not_report_success_metrics(self):
        output = io.StringIO()
        with patch("run.time.perf_counter", side_effect=[1.0, 2.0]), \
             patch("run.subprocess.call", return_value=1), \
             redirect_stderr(output):
            result = run.run_inference(["llama-cli"], Path("."), interactive=False)
        self.assertEqual(result, 1)
        self.assertNotIn("Response time:", output.getvalue())

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown model"):
            run.resolve_model(run.load_catalog(), "does-not-exist")

    def test_no_model_shows_help_without_downloads(self):
        with patch("run.ensure_runtime", side_effect=AssertionError("network attempted")), \
             patch.object(sys, "argv", ["run.py"]):
            self.assertEqual(run.main(), 0)

    def test_arbitrary_catalog_model_uses_same_runner(self):
        model = {
            "name": "Future test model",
            "type": "text",
            "parameters": "200M",
        }
        with patch("run.load_catalog", return_value={"future-200m": model}), \
             patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
             patch("run.ensure_model", return_value=Path("/tmp/future.gguf")), \
             patch("run.subprocess.call", return_value=0) as call, \
             patch.object(sys, "argv", [
                 "run.py", "future-200m", "--max-tokens", "24",
                 "Tell", "me", "something",
             ]):
            self.assertEqual(run.main(), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], str(Path("/tmp/future.gguf")))
        self.assertEqual(command[command.index("--prompt") + 1], "Tell me something")
        self.assertEqual(command[command.index("--predict") + 1], "24")

    def test_legacy_model_flag_still_works(self):
        model = {"name": "Granite", "type": "text", "parameters": "350M",
                 "aliases": ["granite"]}
        with patch("run.load_catalog", return_value={"granite-350m": model}), \
             patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
             patch("run.ensure_model", return_value=Path("/tmp/granite.gguf")), \
             patch("run.subprocess.call", return_value=0) as call, \
             patch.object(sys, "argv", ["run.py", "--model", "granite", "Hello"]):
            self.assertEqual(run.main(), 0)
        self.assertEqual(
            call.call_args.args[0][call.call_args.args[0].index("--prompt") + 1],
            "Hello"
        )

    def test_vision_model_supports_projector_and_image(self):
        model = {
            "name": "Future vision model",
            "type": "vision",
            "parameters": "256M",
            "mmproj": {"filename": "test-mmproj.gguf",
                       "sha256": "0" * 64,
                       "url": "https://example.org/test-mmproj.gguf"},
        }
        with tempfile.TemporaryDirectory() as temp:
            photo = Path(temp) / "test.png"
            photo.write_bytes(b"test")
            with patch("run.load_catalog", return_value={"future-vlm": model}), \
                 patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
                 patch("run.ensure_model", return_value=Path("/tmp/vlm.gguf")), \
                 patch("run.ensure_artifact", return_value=Path("/tmp/mmproj.gguf")) as artifact, \
                 patch("run.subprocess.call", return_value=0) as call, \
                 patch.object(sys, "argv", [
                     "run.py", "future-vlm", "Describe this", "--image", str(photo)
                 ]):
                self.assertEqual(run.main(), 0)
            command = call.call_args.args[0]
            artifact.assert_called_once()
            self.assertEqual(command[command.index("--mmproj") + 1], str(Path("/tmp/mmproj.gguf")))
            self.assertEqual(command[command.index("--image") + 1], str(photo.resolve()))
            self.assertIn("--no-mmproj-offload", command)
            self.assertEqual(command[command.index("--gpu-layers") + 1], "0")

    def test_text_model_rejects_image(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "img.png"
            image.write_bytes(b"x")
            with patch.object(sys, "argv", [
                "run.py", "granite-350m", "Describe", "--image", str(image)
            ]), patch("run.ensure_runtime", side_effect=AssertionError("runtime started")):
                with self.assertRaises(SystemExit) as error:
                    run.main()
                self.assertEqual(error.exception.code, 2)

    def test_catalog_rejects_vision_without_projector(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / "models.json"
            catalog.write_text(
                json.dumps({"bad-vlm": {"name": "No projector",
                                        "type": "vision", "runtime": "llama.cpp"}}),
                encoding="utf-8"
            )
            with patch.object(run, "CATALOG", catalog):
                with self.assertRaisesRegex(RuntimeError, "requires an mmproj"):
                    run.load_catalog()

    def test_models_lists_every_model_and_copy_paste_commands(self):
        output = io.StringIO()
        with patch("run.ensure_runtime", side_effect=AssertionError("network attempted")), \
             patch("sys.stdout", output), \
             patch.object(sys, "argv", ["run.py", "--models"]):
            self.assertEqual(run.main(), 0)
        listing = output.getvalue()
        for model_id in (
            "granite-350m", "smollm2-135m", "smollm2-360m",
            "smolvlm2-256m", "qwen2.5-0.5b"
        ):
            self.assertIn("python run.py {}".format(model_id), listing)
        self.assertIn("smolvlm2-256m", listing)
        self.assertIn("--image benchmark_image/example.png", listing)
        self.assertIn("Text models:", listing)
        self.assertIn("Vision models:", listing)

    def test_vision_benchmark_image_exists_and_is_png(self):
        image = Path(__file__).resolve().parents[1] / "benchmark_image" / "example.png"
        self.assertTrue(image.is_file())
        self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_moondream_is_runnable_but_unsupported_0_5b_is_not_listed(self):
        models = run.load_catalog()
        self.assertNotIn("moondream-0.5b", models)
        self.assertEqual(models["moondream-2b"]["type"], "vision")
        self.assertNotEqual(models["moondream-2b"].get("status"), "research")
        output = io.StringIO()
        with patch("run.ensure_runtime", side_effect=AssertionError("runtime invoked")), \
             patch("sys.stdout", output), \
             patch.object(sys, "argv", ["run.py", "--models"]):
            self.assertEqual(run.main(), 0)
        listing = output.getvalue()
        self.assertNotIn("moondream-0.5b", listing)
        self.assertIn("Run: python run.py moondream-2b", listing)
        self.assertIn("--image benchmark_image/example.png", listing)

    def test_moondream_uses_legacy_template_for_image_inference(self):
        models = run.load_catalog()
        model = models["moondream-2b"]
        self.assertTrue(model["disable_jinja"])
        with tempfile.TemporaryDirectory() as temp:
            photo = Path(temp) / "photo.png"
            photo.write_bytes(b"image")
            with patch("run.ensure_runtime", return_value=Path("/tmp/llama-cli")), \
                 patch("run.ensure_model", return_value=Path("/tmp/moondream.gguf")), \
                 patch("run.ensure_artifact", return_value=Path("/tmp/mmproj.gguf")), \
                 patch("run.subprocess.call", return_value=0) as call, \
                 patch.object(sys, "argv", [
                     "run.py", "moondream-2b", "Describe this image", "--image", str(photo)
                 ]):
                self.assertEqual(run.main(), 0)
            command = call.call_args.args[0]
            self.assertIn("--no-jinja", command)
            self.assertIn("--mmproj", command)
            self.assertIn("--image", command)
            self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
            self.assertNotIn("--chat-template", command)

    def test_moondream_2b_has_matching_gguf_and_projector(self):
        model = run.load_catalog()["moondream-2b"]
        self.assertEqual(model["type"], "vision")
        self.assertEqual(model["runtime"], "llama.cpp")
        self.assertEqual(len(model["sha256"]), 64)
        self.assertEqual(len(model["mmproj"]["sha256"]), 64)
        self.assertIn("20250414", model["mmproj"]["filename"])
        self.assertIn("vicuna", model["filename"])

    def test_model_checksums_and_direct_urls(self):
        models = run.load_catalog()
        for model_id, model in models.items():
            with self.subTest(model=model_id):
                if model.get("status") == "research":
                    self.assertTrue(model.get("model_page", model.get("url", "")).startswith("https://"))
                    continue
                self.assertEqual(len(model["sha256"]), 64)
                self.assertTrue(model["url"].startswith("https://"))
                if model["type"] == "vision":
                    self.assertEqual(len(model["mmproj"]["sha256"]), 64)
                    self.assertTrue(model["mmproj"]["url"].startswith("https://"))

    def test_no_network_for_list(self):
        with patch("run.ensure_runtime", side_effect=AssertionError("network attempted")), \
             patch.object(sys, "argv", ["run.py", "--list"]):
            self.assertEqual(run.main(), 0)


if __name__ == "__main__":
    unittest.main()
