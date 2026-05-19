"""Single control point for all robot audio output.

Design overview
- There is a single output point to the robot: `mini.media.push_audio_sample`.
- Audio comes from multiple sources (LLM conversation voice, emotion sounds from emotion library .wav file, future sources).
- Mixing strategy is additive (fusion): samples from all active sources are summed and clipped to [-1, 1].

Threading model
- A dedicated worker thread runs a time-driven mix loop (~32 ms ticks) and is the
  sole caller of push_audio_sample.
- External sources communicate via thread-safe entry points:
    frames – queue_audio_frame() stages incoming audio frames into the worker queue.
    clips  – queue_audio_clip() loads a .wav clip for additive mixing.
- Each tick the worker drains all queued frames into an internal sample buffer,
  slices out one 32 ms chunk (zero-padded if the buffer is short), fuses in the
  current clip position, and pushes the result. When the frame buffer is empty
  and a clip is active, zeros are fused with the clip so it continues playing.

Safety
- The worker thread is the sole caller of push_audio_sample, preventing
  any collision between sources.
- push_audio_sample is never called while holding the clip lock.
"""

from __future__ import annotations
import time
import logging
import threading
from queue import Empty, Queue
from typing import cast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini


logger = logging.getLogger(__name__)

WORKER_IDLE_TIMEOUT: float = 0.005  # 5 ms idle sleep when no audio is active


class AudioManager:
    """Single output point for all robot audio.

    Sources
    -------
    frames – queue_audio_frame() stages a continuous audio frame into the worker queue.
    clips  – queue_audio_clip() loads and queues a .wav clip for additive mixing. Only .wav files are currently supported.

    Future sources can be added with a new entry point and behavior without changing existing callers.
    """

    def __init__(self, current_robot: ReachyMini):
        """Initialize audio manager."""
        self.current_robot = current_robot

        # Audio queue
        self.frame_queue: Queue[NDArray[np.float32]] = Queue()

        # Clip source state (guarded by _clip_lock)
        self._clip_lock = threading.Lock()
        self._clip_samples: NDArray[np.float32] | None = None
        self._clip_pos: int = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Cached robot output sample rate (resolved once on first use)
        self._output_sample_rate: int | None = None

    def _fuse_clip(self, audio_frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Sum clip samples into the audio frame at the current position of the clip."""
        with self._clip_lock:
            if self._clip_samples is None or self._clip_pos >= len(self._clip_samples):
                self._clip_samples = None
                return audio_frame
            n = len(audio_frame)
            end = min(self._clip_pos + n, len(self._clip_samples))
            clip_chunk = self._clip_samples[self._clip_pos : end]
            mixed = audio_frame.copy()
            mix_len = len(clip_chunk)
            mixed[:mix_len] = np.clip(mixed[:mix_len] + clip_chunk, -1.0, 1.0)
            self._clip_pos = end
            if self._clip_pos >= len(self._clip_samples):
                self._clip_samples = None
            return mixed

    def _get_output_sample_rate(self) -> int:
        if self._output_sample_rate is None:
            self._output_sample_rate = int(self.current_robot.media.get_output_audio_samplerate())
        return self._output_sample_rate

    def _load_clip_and_resample(self, wav_path: Path) -> NDArray[np.float32] | None:
        """Load a clip (e.g. .wav) file and resample to the robot output sample rate."""
        try:
            import scipy.io.wavfile as wavfile
        except ImportError:
            logger.warning("scipy unavailable; cannot load audio clip from %s", wav_path)
            return None

        try:
            file_rate, data = wavfile.read(str(wav_path))
        except Exception as exc:
            logger.warning("Failed to read audio clip %s: %s", wav_path, exc)
            return None

        # Normalise integer PCM to float32 [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        # Collapse to mono
        if data.ndim > 1:
            data = data[:, 0]

        out_rate = self._get_output_sample_rate()
        if file_rate != out_rate and len(data) > 0:
            from scipy.signal import resample

            n_out = int(len(data) * out_rate / file_rate)
            if n_out == 0:
                return None
            data = np.asarray(resample(data, n_out), dtype=np.float32)

        return cast(NDArray[np.float32], data)

    def _push_to_daemon(self, audio_frame: NDArray[np.float32]) -> None:
        """Push audio frame via the single push_audio_sample call site."""
        try:
            self.current_robot.media.push_audio_sample(audio_frame)
        except Exception as exc:
            logger.debug("push_audio_sample failed: %s", exc)

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Audio worker already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True, name="audio-worker-thread")
        self._thread.start()
        logger.debug("Audio worker started")

    def stop(self) -> None:
        """Stop the worker thread."""
        if self._thread is None or not self._thread.is_alive():
            logger.debug("Audio worker not running; stop() ignored")
            return

        logger.info("Stopping audio manager...")

        self.clear_frame_queue()
        self.stop_clip()

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Audio worker stopped")

    def queue_audio_frame(self, audio_frame: NDArray[np.float32]) -> None:
        """Stage a continuous audio frame for the worker to mix and push."""
        self.frame_queue.put(audio_frame.reshape(-1))

    def clear_frame_queue(self) -> None:
        """Discard all pending audio frames."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

    def queue_audio_clip(self, wav_path: Path) -> None:
        """Load and queue a .wav clip to mix into the audio output."""
        samples = self._load_clip_and_resample(wav_path)
        if samples is None:
            return
        with self._clip_lock:
            self._clip_samples = samples
            self._clip_pos = 0
        logger.info("Queued audio clip: %s (%d samples)", wav_path.name, len(samples))

    def stop_clip(self) -> None:
        """Immediately stop the current audio clip."""
        with self._clip_lock:
            self._clip_samples = None
            self._clip_pos = 0
        logger.debug("Audio clip stopped")

    def working_loop(self) -> None:
        """Time-driven mixer: fixed 32 ms chunks at a steady rate, mixing all active sources."""
        out_rate = self._get_output_sample_rate()
        chunk_samples = max(1, int(out_rate * 0.032))
        chunk_duration = chunk_samples / out_rate

        buf: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        t_next = time.monotonic()

        while not self._stop_event.is_set():
            drained = []
            while True:
                try:
                    drained.append(self.frame_queue.get_nowait())
                except Empty:
                    break
            if drained:
                buf = np.concatenate([buf, *drained])

            with self._clip_lock:
                clip_live = self._clip_samples is not None

            if len(buf) == 0 and not clip_live:
                time.sleep(WORKER_IDLE_TIMEOUT)
                t_next = time.monotonic()
                continue

            if len(buf) >= chunk_samples:
                chunk, buf = buf[:chunk_samples], buf[chunk_samples:]
            else:
                chunk = np.zeros(chunk_samples, dtype=np.float32)
                chunk[: len(buf)] = buf
                buf = np.empty(0, dtype=np.float32)

            self._push_to_daemon(self._fuse_clip(chunk))

            t_next += chunk_duration
            wait = t_next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                t_next = time.monotonic()
