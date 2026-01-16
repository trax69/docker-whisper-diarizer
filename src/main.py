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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

class FileHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            if not filename.startswith('.'): 
                logger.info(f"New file detected: {filename}")
                self.queue.put(event.src_path)

def worker(queue: Queue, engine: DiarizationEngine):
    while True:
        original_path = queue.get()
        filename = os.path.basename(original_path)
        
        try:
            proc_path = os.path.join(PROCESSING_DIR, filename)
            
            time.sleep(2)
            
            if not os.path.exists(original_path):
                queue.task_done()
                continue

            shutil.move(original_path, proc_path)
            
            segments = engine.process_file(proc_path)
            
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(OUTPUT_DIR, txt_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.speaker}: {seg.text.strip()}\n")
            
            final_path = os.path.join(COMPLETED_DIR, filename)
            shutil.move(proc_path, final_path)
            logger.info(f"Completed: {filename} -> {txt_filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            if os.path.exists(proc_path):
                error_path = os.path.join(PROCESSING_DIR, f"ERROR_{filename}")
                shutil.move(proc_path, error_path)
        finally:
            queue.task_done()

def ensure_dirs():
    for d in [INPUT_DIR, PROCESSING_DIR, OUTPUT_DIR, COMPLETED_DIR]:
        os.makedirs(d, exist_ok=True)

def main():
    ensure_dirs()
    
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
        if os.path.isfile(full_path) and not f.startswith('.'):
            file_queue.put(full_path)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()