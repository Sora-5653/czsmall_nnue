# SPDX-License-Identifier: MIT
"""Unit tests for the GPU protocol batching and child-process plumbing."""

from __future__ import annotations

import json
import queue
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
import torch

try:
    from . import auto_improve
    from . import gpu_match
    from . import gpu_arena
    from . import iterate
except ImportError:  # pragma: no cover - supports direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import auto_improve  # type: ignore
    import gpu_match  # type: ignore
    import gpu_arena  # type: ignore
    import iterate  # type: ignore


def _request(
    index: int,
    model_id: int,
    *,
    count: int = 1,
    token_slots: int = 2,
    action_slots: int = 3,
    valid_actions: int | None = None,
) -> gpu_match.GpuRequest:
    """Create a deterministic request fixture with identifiable rows."""
    token_np = np.full(
        (count, token_slots, gpu_match.TOKEN_FEATURES),
        float(index),
        dtype=np.float32,
    )
    token_mask_np = np.ones((count, token_slots), dtype=np.float32)
    action_np = np.full(
        (count, action_slots, gpu_match.ACTION_FEATURES),
        float(index) + 0.5,
        dtype=np.float32,
    )
    action_mask_np = np.ones((count, action_slots), dtype=np.float32)
    if valid_actions is not None:
        if not 0 <= valid_actions <= action_slots:
            raise ValueError("valid_actions must fit inside action_slots")
        action_mask_np[:, valid_actions:] = 0.0
    return gpu_match.GpuRequest(
        index,
        None,  # The request fixture does not read from a child process.
        model_id,
        token_np,
        token_mask_np,
        action_np,
        action_mask_np,
    )


class _CaptureStdin:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, payload: bytes) -> int:
        self.data.extend(payload)
        return len(payload)

    def flush(self) -> None:
        return None


class _CaptureProcess:
    def __init__(self) -> None:
        self.stdin = _CaptureStdin()


class _RecordingBatcher(gpu_match.StreamingInferenceQueue):
    """Queue fixture that records jobs without invoking a neural network."""

    def __init__(self) -> None:
        super().__init__(
            torch.nn.Identity(),
            torch.device("cpu"),
            "fp32",
            target_positions=1,
            window_ms=0.0,
        )
        self.calls: list[list[gpu_match.GpuRequest]] = []
        self._calls_lock = threading.Lock()

    def serve(
        self,
        pending: list[gpu_match.GpuRequest],
        stream: torch.cuda.Stream | None = None,
    ) -> list[tuple[list[gpu_match.GpuRequest], float]]:
        del stream
        with self._calls_lock:
            self.calls.append(list(pending))
        return []


class GpuPipelineTests(unittest.TestCase):
    def test_auto_improve_tracks_arena_sidecar(self) -> None:
        artifacts = auto_improve.generation_artifacts(
            Path("data"), Path("models"), 7
        )

        self.assertIn(Path("models/gen7.arena.json"), artifacts)

    def test_compatible_groups_preserve_first_seen_model_and_request_order(self) -> None:
        pending = [
            _request(10, 7),
            _request(11, 3),
            _request(12, 7),
            _request(13, 9),
            _request(14, 3),
        ]

        groups = gpu_match.StreamingInferenceQueue.compatible_groups(pending)

        self.assertEqual(
            [[request.model_id for request in group] for group in groups],
            [[7, 7], [3, 3], [9]],
        )
        self.assertEqual(
            [[request.index for request in group] for group in groups],
            [[10, 12], [11, 14], [13]],
        )

    def test_async_dispatcher_preserves_collected_batch_and_closes(self) -> None:
        batcher = _RecordingBatcher()
        dispatcher = gpu_match.AsyncStreamingInferenceDispatcher(
            batcher,
            queue.Queue(),
            max_queued_batches=2,
            gpu_workers=2,
        )
        pending = [
            _request(20, 1),
            _request(21, 0),
            _request(22, 1),
            _request(23, 0),
        ]

        try:
            dispatcher.submit(pending)
        finally:
            dispatcher.close()

        self.assertEqual(
            [[request.index for request in group] for group in batcher.calls],
            [[20, 21, 22, 23]],
        )
        self.assertEqual(len(batcher.calls), 1)
        self.assertTrue(all(not thread.is_alive() for thread in dispatcher._threads))

    def test_stderr_drainer_consumes_output_larger_than_pipe_buffer(self) -> None:
        payload_size = 1024 * 1024
        script = (
            "import sys\n"
            f"sys.stderr.buffer.write(b'e' * {payload_size})\n"
            "sys.stderr.flush()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        drainer = None
        try:
            drainer = gpu_match.ChildStderrDrainer(proc, "test-stderr-drainer")
            self.assertEqual(proc.wait(timeout=20), 0)
            self.assertEqual(drainer.finish(timeout=5), "e" * payload_size)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if drainer is not None:
                drainer.finish(timeout=5)
            if proc.stderr is not None and not proc.stderr.closed:
                proc.stderr.close()

    def test_request_assembly_pads_rectangular_edges_and_keeps_offsets(self) -> None:
        first = _request(30, 0, count=2, token_slots=2, action_slots=3)
        second = _request(40, 0, count=1, token_slots=4, action_slots=1)

        tokens, token_mask, actions, action_mask, offsets = (
            gpu_match._assemble_request_batch([first, second])
        )

        self.assertEqual(tokens.shape, (3, 4, gpu_match.TOKEN_FEATURES))
        self.assertEqual(token_mask.shape, (3, 4))
        self.assertEqual(actions.shape, (3, 3, gpu_match.ACTION_FEATURES))
        self.assertEqual(action_mask.shape, (3, 3))
        self.assertEqual(
            [(request.index, start, count) for request, start, count in offsets],
            [(30, 0, 2), (40, 2, 1)],
        )
        np.testing.assert_array_equal(tokens[:2, :2], first.token_np)
        np.testing.assert_array_equal(token_mask[:2, :2], first.token_mask_np)
        np.testing.assert_array_equal(tokens[2, :4], second.token_np[0])
        np.testing.assert_array_equal(token_mask[2, :4], second.token_mask_np[0])
        self.assertTrue(np.all(tokens[:2, 2:] == 0.0))
        self.assertTrue(np.all(token_mask[:2, 2:] == 0.0))
        np.testing.assert_array_equal(actions[:2, :3], first.action_np)
        np.testing.assert_array_equal(action_mask[:2, :3], first.action_mask_np)
        np.testing.assert_array_equal(actions[2, :1], second.action_np[0])
        np.testing.assert_array_equal(action_mask[2, :1], second.action_mask_np[0])
        self.assertTrue(np.all(actions[2, 1:] == 0.0))
        self.assertTrue(np.all(action_mask[2, 1:] == 0.0))

    def test_single_request_assembly_reuses_owned_arrays(self) -> None:
        request = _request(45, 0, count=2, token_slots=4, action_slots=5)

        tokens, token_mask, actions, action_mask, offsets = (
            gpu_match._assemble_request_batch([request])
        )

        self.assertIs(tokens, request.token_np)
        self.assertIs(token_mask, request.token_mask_np)
        self.assertIs(actions, request.action_np)
        self.assertIs(action_mask, request.action_mask_np)
        self.assertEqual(offsets, [(request, 0, 2)])

    def test_request_assembly_rejects_empty_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty GPU request batch"):
            gpu_match._assemble_request_batch([])

    def test_response_framing_preserves_request_and_action_boundaries(self) -> None:
        first_proc = _CaptureProcess()
        second_proc = _CaptureProcess()
        first = _request(
            50,
            0,
            count=2,
            token_slots=2,
            action_slots=3,
            valid_actions=1,
        )
        second = _request(
            60,
            0,
            count=1,
            token_slots=3,
            action_slots=2,
            valid_actions=2,
        )
        first.proc = first_proc
        second.proc = second_proc

        gpu_match._serve_request_batch(
            {0: object()},
            torch.device("cpu"),
            [first, second],
            "fp32",
            diagnostic_dummy=True,
        )

        for process, expected_actions, expected_count in (
            (first_proc, [1, 1], 2),
            (second_proc, [2], 1),
        ):
            wire = bytes(process.stdin.data)
            self.assertGreaterEqual(len(wire), 8)
            self.assertEqual(
                struct.unpack_from("<4sI", wire, 0),
                (b"TGPR", expected_count),
            )
            cursor = 8
            for action_count in expected_actions:
                (actual_actions,) = struct.unpack_from("<I", wire, cursor)
                cursor += 4
                self.assertEqual(actual_actions, action_count)
                policy = np.frombuffer(
                    wire, dtype="<f4", count=actual_actions, offset=cursor
                )
                cursor += 4 * actual_actions
                wdl = np.frombuffer(wire, dtype="<f4", count=3, offset=cursor)
                cursor += 12
                aux = np.frombuffer(wire, dtype="<f4", count=4, offset=cursor)
                cursor += 16
                np.testing.assert_allclose(policy, 1.0 / action_count)
                np.testing.assert_allclose(wdl, [0.0, 1.0, 0.0])
                np.testing.assert_allclose(aux, np.zeros(4, dtype=np.float32))
            self.assertEqual(cursor, len(wire))

    def test_arena_protocol_command_normalizes_and_formats_arguments(self) -> None:
        protocol = gpu_arena.ArenaProtocolConfig(
            sims=0,
            pieces=0,
            batch_size=0,
            determinizations=0,
            use_gumbel=True,
            candidate_sims=-7,
            champion_sims=-8,
            candidate_gumbel=-9,
            champion_gumbel=-10,
            gumbel_c_scale=0.125,
            gumbel_noise_scale=0.25,
            candidate_timing_actions=-11,
            champion_timing_actions=-12,
            candidate_gumbel_noise_scale=0.75,
            champion_gumbel_noise_scale=0.5,
        )

        command = protocol.command("engine.exe", pairs=0, seed=(1 << 64) + 5)

        self.assertEqual(
            command,
            [
                "engine.exe",
                "gpu-arena-protocol",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "5",
                "-7",
                "-8",
                "-9",
                "-10",
                "0.125",
                "0.25",
                "-11",
                "-12",
                "0.75",
                "0.5",
            ],
        )

    def test_result_json_is_atomic_and_rejects_nonfinite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arena.json"
            path.write_text('{"old": true}\n', encoding="utf-8")

            gpu_arena._write_result_json(path, {"schema_version": 1, "value": 2.5})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"schema_version": 1, "value": 2.5},
            )
            self.assertEqual(list(root.glob("*.tmp")), [])

            with self.assertRaises(ValueError):
                gpu_arena._write_result_json(path, {"value": float("nan")})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"schema_version": 1, "value": 2.5},
            )
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_load_gpu_arena_result_validates_schema_and_required_fields(self) -> None:
        result = {
            "games_played": 8,
            "candidate_wins": 4,
            "champion_wins": 3,
            "draws": 1,
            "win_rate": 0.5625,
            "ci_lower": 0.2,
            "ci_upper": 0.8,
            "promoted": False,
            "extra_telemetry": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena.json"
            path.write_text(
                json.dumps({"schema_version": 1, "result": result}),
                encoding="utf-8",
            )
            self.assertEqual(iterate.load_gpu_arena_result(path), result)

            path.write_text(
                json.dumps({"schema_version": 2, "result": result}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "unsupported GPU Arena result schema"
            ):
                iterate.load_gpu_arena_result(path)

            path.write_text(
                json.dumps({"schema_version": 1, "result": {"promoted": False}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                iterate.load_gpu_arena_result(path)

            invalid = dict(result)
            invalid["promoted"] = 1
            path.write_text(
                json.dumps({"schema_version": 1, "result": invalid}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "promoted field is not boolean"
            ):
                iterate.load_gpu_arena_result(path)

            invalid = dict(result)
            invalid["games_played"] = 9
            path.write_text(
                json.dumps({"schema_version": 1, "result": invalid}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "do not sum to total"):
                iterate.load_gpu_arena_result(path)

            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "root is not an object"):
                iterate.load_gpu_arena_result(path)

            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot read GPU Arena result"):
                iterate.load_gpu_arena_result(path)


if __name__ == "__main__":
    unittest.main()
