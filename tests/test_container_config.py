"""M1 / M2 – guard-rails on the container build configuration.

Dockerfile and docker-compose.yml are not copied into the image, so those
checks are skipped when the suite runs inside the container.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(name: str) -> str:
    return os.path.join(ROOT, name)


def _read(name: str) -> str:
    with open(_path(name), encoding="utf-8") as fh:
        return fh.read()


class RequirementsTest(unittest.TestCase):
    def test_pyannote_is_pinned_below_4(self):
        # pyannote.audio 4.x changed the pipeline API (token=, exclusive diarization…).
        line = next(l for l in _read("requirements.txt").splitlines()
                    if l.strip().startswith("pyannote.audio"))
        self.assertRegex(line, r"<\s*4(\.0)*\b")
        self.assertRegex(line, r">=\s*3\.1")


@unittest.skipUnless(os.path.exists(_path("Dockerfile")), "Dockerfile not available")
class DockerfileTest(unittest.TestCase):
    def test_pyannote_cache_lives_in_the_mounted_hf_cache_volume(self):
        match = re.search(r"(?m)^ENV\s+PYANNOTE_CACHE=(\S+)", _read("Dockerfile"))
        self.assertIsNotNone(match, "Dockerfile must set PYANNOTE_CACHE")
        self.assertTrue(match.group(1).startswith("/root/.cache/huggingface/"))

    def test_tests_are_copied_into_the_image(self):
        self.assertTrue(re.search(r"(?m)^COPY\s+tests/\s+/app/tests/", _read("Dockerfile")),
                        "Dockerfile must COPY tests/ /app/tests/")


@unittest.skipUnless(os.path.exists(_path("docker-compose.yml")), "compose file not available")
class ComposeTest(unittest.TestCase):
    def test_hf_cache_volume_is_mounted(self):
        self.assertTrue(":/root/.cache/huggingface" in _read("docker-compose.yml"))

    def test_daemon_exposes_the_new_tuning_variables(self):
        compose = _read("docker-compose.yml")
        for var in ("NUM_SPEAKERS", "MIN_SPEAKERS", "MAX_SPEAKERS", "MIN_TURN_WORDS",
                    "MIN_TURN_SEC", "MAIN_SPEAKER_LABEL", "PAUSE_SPLIT_SEC"):
            with self.subTest(var=var):
                self.assertTrue(f"{var}=${{{var}" in compose, f"{var} missing in compose")


if __name__ == "__main__":
    unittest.main()
