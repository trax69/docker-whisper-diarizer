"""M3 – decide which files dropped in the input folder are audio/video to transcribe.

An allow-list of extensions is used instead of a deny-list so subtitles (.vtt,
.srt), transcripts (.txt), partial downloads (.part, .crdownload) or any other
stray file are never sent to Whisper.
"""
import os

MEDIA_EXTENSIONS = frozenset({
    # video
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".mpg", ".mpeg",
    ".ts", ".mts", ".3gp",
    # audio
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".amr", ".mka",
})

# Files the daemon itself creates next to the media file while processing it.
AUXILIARY_SUFFIXES = (".words.json", ".diarize.wav", ".diarize.json", ".tmp.wav", ".retry")


def is_media_file(filename: str) -> bool:
    name = os.path.basename(filename)
    lower = name.lower()
    if name.startswith(".") or name.startswith("ERROR_"):
        return False
    if lower.endswith(AUXILIARY_SUFFIXES):  # checked first: ".diarize.wav" ends in ".wav"
        return False
    return os.path.splitext(lower)[1] in MEDIA_EXTENSIONS
