"""M3 – the folder watcher must only queue real audio/video files."""
import unittest

from media_filter import AUXILIARY_SUFFIXES, is_media_file


class IsMediaFileTest(unittest.TestCase):
    def test_accepts_common_audio_and_video_files(self):
        for name in ("631009.mp4", "clase.m4a", "audio.mp3", "voz.wav", "VIDEO.MKV",
                     "grabacion.webm", "nota.ogg"):
            with self.subTest(name=name):
                self.assertTrue(is_media_file(name))

    def test_rejects_subtitles_and_transcripts(self):
        for name in ("631009.vtt", "631009.srt", "631009.txt", "631009.VTT", "notas.md",
                     "datos.json"):
            with self.subTest(name=name):
                self.assertFalse(is_media_file(name))

    def test_rejects_incomplete_downloads(self):
        for name in ("video.mp4.part", "video.mp4.crdownload", "video.tmp"):
            with self.subTest(name=name):
                self.assertFalse(is_media_file(name))

    def test_rejects_hidden_and_error_files(self):
        self.assertFalse(is_media_file(".video.mp4"))
        self.assertFalse(is_media_file("ERROR_video.mp4"))

    def test_rejects_auxiliary_files_created_while_processing(self):
        for suffix in AUXILIARY_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertFalse(is_media_file("video.mp4" + suffix))

    def test_auxiliary_suffixes_include_the_diarization_json(self):
        # Left behind if the container dies mid-diarization; previously it was
        # neither cleaned up nor ignored, so it was re-queued as "media".
        self.assertIn(".diarize.json", AUXILIARY_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
