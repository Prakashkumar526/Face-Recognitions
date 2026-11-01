"""Step 4 - view attendance.   python view_attendance.py [--date DD-MM-YYYY] [--all] [--csv out.csv]"""
import argparse
import csv
from datetime import datetime

from attendance import DATE_FMT, HEADERS, AttendanceLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime(DATE_FMT), help="default: today")
    ap.add_argument("--all", action="store_true", help="show every date")
    ap.add_argument("--csv", help="also export the shown rows to this CSV file")
    a = ap.parse_args()

    rows = AttendanceLog().records(None if a.all else a.date)
    print(f"{HEADERS[0]:<18}{HEADERS[1]:<12}{HEADERS[2]}")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:<12}{r[2]}")
    print(f"\n{len(rows)} record(s)" + ("" if a.all else f" for {a.date}"))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADERS)
            w.writerows(rows)


if __name__ == "__main__":
    main()
