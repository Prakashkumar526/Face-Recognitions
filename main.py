"""Menu launcher:  python main.py"""
import datacollect
import encode_faces
import recognize
import view_attendance

MENU = """
=== Facial Recognition Attendance System ===
 1. Register a person (capture dataset)
 2. Train / encode faces
 3. Start attendance (webcam)
 4. View today's attendance
 5. Quit
"""

ACTIONS = {
    "1": lambda: datacollect.collect(input("Enter Roll Number: ")),
    "2": encode_faces.main,
    "3": recognize.run,
    "4": view_attendance.main,
}

if __name__ == "__main__":
    import sys
    while True:
        print(MENU)
        choice = input("Choose: ").strip()
        if choice == "5":
            break
        action = ACTIONS.get(choice)
        if not action:
            continue
        sys.argv = sys.argv[:1]          # menu items must not see leftover CLI args
        try:
            action()
        except (SystemExit, Exception) as e:   # keep the menu alive on errors
            print(f"[!] {e}")
