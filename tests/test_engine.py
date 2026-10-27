"""Integration of DiarizationEngine with the injected collaborators.

No GPU, Whisper or pyannote needed: heavy steps are replaced by test doubles.
"""
import json
import unittest
from unittest import mock

import engine as engine_module
from engine import DiarizationEngine
from speaker_count import SpeakerCountHint
from transcript_models import Segment, SpeakerTurn
from tests.helpers import word


class _RecordingAlignment:
    def __init__(self):
        self.calls = []

    def run(self, words, turns):
        self.calls.append((words, turns))
        return [Segment(0, 1, "hola", "SPEAKER_00")]


def _make_engine(alignment=None, hint=None):
    return DiarizationEngine(model_size="tiny", device="cpu",
                             speaker_hint=hint, alignment=alignment or _RecordingAlignment())


class ProcessFileTest(unittest.TestCase):
    def test_delegates_words_and_turns_to_the_alignment_pipeline(self):
        alignment = _RecordingAlignment()
        eng = _make_engine(alignment)
        words, turns = [word(0, 1, "hola")], [SpeakerTurn(0, 1, "A")]
        with mock.patch.object(eng, "_transcribe_with_cache", return_value=words), \
                mock.patch.object(eng, "_diarize", return_value=turns):
            result = eng.process_file("video.mp4")
        self.assertEqual(alignment.calls, [(words, turns)])
        self.assertEqual(result, [Segment(0, 1, "hola", "SPEAKER_00")])

    def test_uses_the_configured_speaker_hint(self):
        hint = SpeakerCountHint(max_speakers=3)
        eng = _make_engine(hint=hint)
        with mock.patch.object(eng, "_transcribe_with_cache", return_value=[word(0, 1)]), \
                mock.patch.object(eng, "_diarize", return_value=[]) as diarize:
            eng.process_file("video.mp4")
        diarize.assert_called_once_with("video.mp4", hint)

    def test_per_call_hint_overrides_the_configured_one(self):
        eng = _make_engine(hint=SpeakerCountHint(max_speakers=3))
        override = SpeakerCountHint(num_speakers=2)
        with mock.patch.object(eng, "_transcribe_with_cache", return_value=[word(0, 1)]), \
                mock.patch.object(eng, "_diarize", return_value=[]) as diarize:
            eng.process_file("video.mp4", speaker_hint=override)
        diarize.assert_called_once_with("video.mp4", override)

    def test_no_speech_skips_diarization(self):
        eng = _make_engine()
        with mock.patch.object(eng, "_transcribe_with_cache", return_value=[]), \
                mock.patch.object(eng, "_diarize") as diarize:
            self.assertEqual(eng.process_file("video.mp4"), [])
        diarize.assert_not_called()

    def test_default_alignment_pipeline_is_built_when_none_is_injected(self):
        eng = DiarizationEngine(model_size="tiny", device="cpu")
        with mock.patch.object(eng, "_transcribe_with_cache", return_value=[word(0, 1, "hola")]), \
                mock.patch.object(eng, "_diarize", return_value=[]):
            self.assertEqual(eng.process_file("video.mp4"),
                             [Segment(0, 1, "hola", "SPEAKER_00")])


class _FakeProcess:
    """Stands in for multiprocessing.Process: runs a fake worker synchronously."""
    launched_args = None

    def __init__(self, target, args):
        self._args = args
        _FakeProcess.launched_args = args
        self.exitcode = None

    def start(self):
        _wav, _kwargs, output_path = self._args
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_03"}], fh)
        self.exitcode = 0

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class DiarizeTest(unittest.TestCase):
    def test_forwards_hint_kwargs_to_the_worker_and_parses_turns(self):
        import os
        import tempfile

        eng = _make_engine()
        fake_ctx = mock.Mock(Process=_FakeProcess)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(DiarizationEngine, "_extract_to_wav"), \
                mock.patch.object(engine_module.mp, "get_context", return_value=fake_ctx):
            turns = eng._diarize(os.path.join(tmp, "video.mp4"),
                                 SpeakerCountHint(min_speakers=1, max_speakers=4))

        self.assertEqual(_FakeProcess.launched_args[1], {"min_speakers": 1, "max_speakers": 4})
        self.assertEqual(turns, [SpeakerTurn(0.0, 1.5, "SPEAKER_03")])


if __name__ == "__main__":
    unittest.main()
