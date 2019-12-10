import face_recognition
import cv2
import numpy as np

# Built of example from https://github.com/ageitgey/face_recognition

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a Obama picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a Cody's picture and learn how to recognize it.
cody_image = face_recognition.load_image_file("cody.jpg")
cody_face_encoding = face_recognition.face_encodings(cody_image)[0]

# Load a Mercy's picture and learn how to recognize it.
mercy_image = face_recognition.load_image_file("mercy.jpeg")
mercy_face_encoding = face_recognition.face_encodings(mercy_image)[0]

# Load a Tom Brady's picture and learn how to recognize it.
brady_image = face_recognition.load_image_file("brady.png")
brady_face_encoding = face_recognition.face_encodings(brady_image)[0]

elon_image = face_recognition.load_image_file("elon.jpeg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

rdjr_image = face_recognition.load_image_file("robertDowny.jpg")
rdjr_face_encoding = face_recognition.face_encodings(rdjr_image)[0]

rodriguez_image = face_recognition.load_image_file("Rodriguez.jpg")
rodriguez_face_encoding = face_recognition.face_encodings(rodriguez_image)[0]

serena_image = face_recognition.load_image_file("Serena.jpg")
serena_face_encoding = face_recognition.face_encodings(serena_image)[0]

trump_image = face_recognition.load_image_file("trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

anne_image = face_recognition.load_image_file("Anne_Hathaway.jpg")
anne_face_encoding = face_recognition.face_encodings(anne_image)[0]

brad_image = face_recognition.load_image_file("BradPitt.jpg")
brad_face_encoding = face_recognition.face_encodings(brad_image)[0]

julia_image = face_recognition.load_image_file("JuliaRoberts.jpg")
julia_face_encoding = face_recognition.face_encodings(julia_image)[0]

matt_image = face_recognition.load_image_file("Matthew-McConaughey.jpg")
matt_face_encoding = face_recognition.face_encodings(matt_image)[0]

patrick_image = face_recognition.load_image_file("PatrickMahomes.jpeg")
patrick_face_encoding = face_recognition.face_encodings(patrick_image)[0]

tom_image = face_recognition.load_image_file("tomhanks.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

will_image = face_recognition.load_image_file("willsmith.jpg")
will_face_encoding = face_recognition.face_encodings(will_image)[0]

ye_image = face_recognition.load_image_file("YeDuan.jpg")
ye_face_encoding = face_recognition.face_encodings(ye_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    cody_face_encoding, 
    mercy_face_encoding,
    brady_face_encoding,
    elon_face_encoding,
    rdjr_face_encoding,
    rodriguez_face_encoding,
    serena_face_encoding,
    trump_face_encoding,
    anne_face_encoding,
    brad_face_encoding,
    julia_face_encoding,
    matt_face_encoding,
    patrick_face_encoding,
    tom_face_encoding,
    will_face_encoding,
    ye_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Cody Polton",
    "Mecry Housh",
    "Tom Brady",
    "Elon Musk",
    "Robert Downey Jr.",
    "Alex Rodriguez",
    "Serena Williams",
    "Donald Trump",
    "Anne Hathaway",
    "Brad Pitt",
    "Julia Roberts",
    "Patrick Mahomes",
    "Tom Hanks",
    "Will Smith",
    "Ye Duan"

]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()