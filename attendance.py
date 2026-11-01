"""Excel attendance log: columns Roll Number | Time | Date (one row per person per day)."""
from datetime import datetime

from openpyxl import Workbook, load_workbook

import config

HEADERS = ["Roll Number", "Time", "Date"]
DATE_FMT = "%d-%m-%Y"
TIME_FMT = "%H:%M:%S"


class AttendanceLog:
    def __init__(self, path=config.ATTENDANCE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            wb = Workbook()
            wb.active.title = "Attendance"
            wb.active.append(HEADERS)
            wb.save(self.path)
        self._seen = {(r, d) for r, _, d in self._rows()}   # cache: avoid re-reading the file per frame

    def _rows(self):
        wb = load_workbook(self.path, read_only=True)
        try:
            rows = list(wb.active.iter_rows(min_row=2, values_only=True))
        finally:
            wb.close()
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows if r and r[0]]

    def already_marked(self, roll, when=None):
        return (str(roll), (when or datetime.now()).strftime(DATE_FMT)) in self._seen

    def mark(self, roll, when=None):
        """Append a row unless `roll` is already marked today. Returns True if written."""
        when = when or datetime.now()
        key = (str(roll), when.strftime(DATE_FMT))
        if key in self._seen:
            return False
        try:
            wb = load_workbook(self.path)
            wb.active.append([str(roll), when.strftime(TIME_FMT), key[1]])
            wb.save(self.path)
        except PermissionError:          # file open in Excel on Windows -> retry on next frame
            print(f"[!] Close {self.path.name} in Excel so attendance can be saved.")
            return False
        self._seen.add(key)
        return True

    def records(self, date=None):
        """All rows, or only those for `date` (a dd-mm-YYYY string)."""
        rows = self._rows()
        return [r for r in rows if date is None or r[2] == date]
