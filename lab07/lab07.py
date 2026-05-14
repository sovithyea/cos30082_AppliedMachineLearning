import cv2
import os
import csv
import face_recognition
from datetime import datetime

FACES_DB = "faces_db"
LOG_FILE = "attendance_log.csv"
THRESHOLD = 0.5
ABSENCE_LIMIT = 1
PROCESS_EVERY = 3 # only process every 3rd frame

known_encodings = []
known_names = []

print("Loading face database...")
for person_name in os.listdir(FACES_DB):
    person_folder = os.path.join(FACES_DB, person_name)
    if not os.path.isdir(person_folder):
        continue
    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"Loaded {len(known_encodings)} face(s) from database.")

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Event", "Time"])

def log_event(name, event):
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, event, timestamp])
    print(f"[LOG] {name} : {event} at {timestamp}")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # lower resolution = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not found!")
    exit()

currently_present = {}
frame_count = 0
face_locations = []
face_encodings_list = []
names_to_display = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run recognition every 3rd frame
    if frame_count % PROCESS_EVERY == 0:
        # Shrink frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings_list = face_recognition.face_encodings(rgb_small, face_locations)

        seen_this_frame = set()
        names_to_display = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings_list):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = face_distances.argmin()

            if matches[best_match_index] and face_distances[best_match_index] < THRESHOLD:
                name = known_names[best_match_index]

            names_to_display.append((top, right, bottom, left, name))

            if name != "Unknown":
                seen_this_frame.add(name)
                if name not in currently_present:
                    currently_present[name] = 0
                    log_event(name, "ENTER")
                else:
                    currently_present[name] = 0

        for name in list(currently_present.keys()):
            if name not in seen_this_frame:
                currently_present[name] += 1
                if currently_present[name] >= ABSENCE_LIMIT:
                    log_event(name, "EXIT")
                    del currently_present[name]

    # Scale back up coordinates (x4 because we shrunk by 0.25)
    for (top, right, bottom, left, name) in names_to_display:
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Facial Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for name in list(currently_present.keys()):
    log_event(name, "EXIT")

cap.release()
cv2.destroyAllWindows()