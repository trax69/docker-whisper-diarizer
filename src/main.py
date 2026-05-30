import signal
import sys
import time
import os
import logging
import shutil
from queue import Queue
from threading import Thread
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from engine import DiarizationEngine

# Configuration
INPUT_DIR = "/app/data/input"
PROCESSING_DIR = "/app/data/processing"
OUTPUT_DIR = "/app/data/output"
COMPLETED_DIR = "/app/data/completed"
MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
MAX_RETRIES = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")


def _is_media_file(filename: str) -> bool:
    skip = ('.words.json', '.diarize.wav', '.tmp.wav', '.retry')
    return (
        not filename.startswith('.')
        and not filename.startswith('ERROR_')
        and not any(filename.endswith(s) for s in skip)
    )


class FileHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            if _is_media_file(filename):
                logger.info(f"New file detected: {filename}")
                self.queue.put(event.src_path)


def _retry_path(proc_path: str) -> str:
    return proc_path + ".retry"


def _get_retry_count(proc_path: str) -> int:
    try:
        retry_file = _retry_path(proc_path)
        if os.path.exists(retry_file):
            with open(retry_file) as f:
                return int(f.read().strip())
    except Exception:
        pass
    return 0


def _increment_retry(proc_path: str) -> int:
    count = _get_retry_count(proc_path) + 1
    try:
        with open(_retry_path(proc_path), 'w') as f:
            f.write(str(count))
    except Exception:
        pass
    return count


def _cleanup_leftovers(proc_path: str) -> None:
    """Delete auxiliary files associated with proc_path (words cache, WAV, retry counter)."""
    for suffix in ('.words.json', '.diarize.wav', '.tmp.wav', '.retry'):
        leftover = proc_path + suffix
        try:
            if os.path.exists(leftover):
                os.unlink(leftover)
                logger.info(f"Cleaned up: {os.path.basename(leftover)}")
        except OSError:
            pass


def _cleanup_orphaned_wavs(directory: str) -> None:
    for f in os.listdir(directory):
        if f.endswith('.tmp.wav') or f.endswith('.diarize.wav'):
            path = os.path.join(directory, f)
            try:
                os.unlink(path)
                logger.info(f"Removed orphaned WAV: {f}")
            except OSError as e:
                logger.warning(f"Could not remove orphaned WAV {f}: {e}")


def _move_to_error(proc_path: str, filename: str) -> None:
    """Clean up leftovers then move media file to ERROR_ prefix."""
    _cleanup_leftovers(proc_path)
    error_path = os.path.join(PROCESSING_DIR, f"ERROR_{filename}")
    if os.path.exists(proc_path):
        shutil.move(proc_path, error_path)


def worker(queue: Queue, engine: DiarizationEngine):
    while True:
        original_path = queue.get()
        filename = os.path.basename(original_path)
        proc_path = os.path.join(PROCESSING_DIR, filename)

        try:
            already_in_processing = os.path.normpath(original_path) == os.path.normpath(proc_path)

            if not already_in_processing:
                time.sleep(2)
                if not os.path.exists(original_path):
                    continue  # finally handles task_done
                shutil.move(original_path, proc_path)
            else:
                retry_count = _increment_retry(proc_path)
                if retry_count > MAX_RETRIES:
                    logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {filename}, moving to error.")
                    _move_to_error(proc_path, filename)
                    continue  # finally handles task_done
                logger.info(f"Resuming stuck file (attempt {retry_count}/{MAX_RETRIES}): {filename}")

            segments = engine.process_file(proc_path)

            txt_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(OUTPUT_DIR, txt_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.speaker}: {seg.text.strip()}\n")

            final_path = os.path.join(COMPLETED_DIR, filename)
            _cleanup_leftovers(proc_path)
            shutil.move(proc_path, final_path)
            logger.info(f"Completed: {filename} -> {txt_filename}")

        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")
            _move_to_error(proc_path, filename)
        finally:
            queue.task_done()  # always called exactly once per item


def ensure_dirs():
    for d in [INPUT_DIR, PROCESSING_DIR, OUTPUT_DIR, COMPLETED_DIR]:
        os.makedirs(d, exist_ok=True)


def main():
    ensure_dirs()
    _cleanup_orphaned_wavs(PROCESSING_DIR)

    engine = DiarizationEngine(model_size=MODEL_SIZE)

    file_queue = Queue()
    processor_thread = Thread(target=worker, args=(file_queue, engine), daemon=True)
    processor_thread.start()

    event_handler = FileHandler(file_queue)
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()

    logger.info(f"Monitoring {INPUT_DIR} for new files (Polling Mode)...")

    for f in os.listdir(INPUT_DIR):
        full_path = os.path.join(INPUT_DIR, f)
        if os.path.isfile(full_path) and _is_media_file(f):
            file_queue.put(full_path)

    for f in os.listdir(PROCESSING_DIR):
        full_path = os.path.join(PROCESSING_DIR, f)
        if os.path.isfile(full_path) and _is_media_file(f):
            logger.info(f"Found stuck file in processing, queuing for recovery: {f}")
            file_queue.put(full_path)

    def _handle_signal(signum, frame):
        signame = signal.Signals(signum).name
        logger.warning(f"Received signal {signame} ({signum}) — shutting down for restart.")
        observer.stop()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while True:
            time.sleep(5)
            if not processor_thread.is_alive():
                logger.critical("Worker thread died — restarting container.")
                observer.stop()
                sys.exit(1)
            if not observer.is_alive():
                logger.warning("File observer died — restarting container.")
                sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected exception in main loop: {e}")
        observer.stop()
        sys.exit(1)
    finally:
        observer.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
