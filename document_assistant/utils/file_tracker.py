import getpass
import os, json, hashlib
from datetime import datetime

TRACKER_PATH = 'results/file_index.json'


class FileTracker:
    """
    Responsilble for
    1. calculating hash
    2. Adding metadata to the json file
       -- file name
       -- hash
       -- added at (date and time)
       -- added by 
       -- size
       -- 
    3. check file is already loaded:: Retrieve metadata (based on filename and hash)
    4. Delete metadata based on hash and filename
    """
    def __init__(self, tracker_path: str=TRACKER_PATH):
        self.tracker_path = tracker_path
        os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
        self.data = self._load_file()
    
    def _load_file(self):
        if os.path.exists(self.tracker_path):
            with open(self.tracker_path, 'r') as file:
                return json.load(file)
        return {}
    
    def _save_file(self):
        with open(self.tracker_path, 'w') as file:
            json.dump(self.data, file, indent=4)

    def hash_file(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_indexed(self, file_path: str) -> bool:
        current_hash = self.hash_file(file_path)
        file_name = os.path.basename(file_path)
        info  = self.data.get(file_name) 
        if info and info['hash'] == current_hash:
            return True
        return False
    
    def add_file(self, file_path: str) :
    #     Adding metadata to the json file
    #    -- file name
    #    -- hash
    #    -- added at (date and time)
    #    -- added by 
    #    -- size
        file_name = os.path.basename(file_path)
        self.data[file_name] = {
            "hash": self.hash_file(file_path=file_path),
            # "num_chunks": num_chunks,
            "added_at": datetime.now().isoformat(),
            "added_by": getpass.getuser(),
            "size_kb": round(os.path.getsize(file_path) / 1024, 2)
        }
        self._save_file()

    def remove_file(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        if self.data.get(file_name):
            del self.data[file_name]
            self._save_file()

    def list_files(self) -> list:
        return list(self.data.keys())
    
    def check_file_loaded(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        return self.data.get(file_name) != None
