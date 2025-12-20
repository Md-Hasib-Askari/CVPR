from datetime import datetime
import os
import csv

class AttendanceManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.marked = set()
        self.records = []

        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])

    def mark(self, name):
        if name in self.marked:
            return False
        
        with open(self.file_path, "a") as f:
            writer = csv.writer(f)

            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time = now.strftime("%H:%M:%S")

            writer.writerow([name, date, time])
        self.marked.add(name)
        self.records.append((name, date, time))
        return True

    def get_attendance(self):
        import pandas as pd
        return pd.read_csv(self.file_path)
    
    def clear(self):
        self.marked.clear()
        with open(self.file_path, "w") as f:
            f.write("Name,Date,Time\n")
