from datetime import datetime
import os

class AttendanceManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.marked = set()

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("Name,Time\n")

    def mark(self, name):
        if name not in self.marked:
            self.marked.add(name)
            with open(self.file_path, "a") as f:
                f.write(f"{name},{datetime.now()}\n")
