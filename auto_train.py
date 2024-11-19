import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('intents.json'):
            print(f'{event.src_path} has been modified. Training the model...')
            subprocess.run(['python', 'scripts/train.py'])
            print('Model trained successfully.')

if __name__ == "__main__":
    path = 'data'  # Directory where intents.json is located
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    print('Monitoring file changes. Press Ctrl+C to stop.')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
