from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './Waiting_room_hospital.mp4'
cap = cv2.VideoCapture(video_path)

#Inicializamos esta variable a True ya que .read() devuelve un bool dependiendo de si ha leido el frame o no.
ret = True
# read frame
while ret:
    ret, frame = cap.read()

    if ret :

        # object detection
        # track objects

        #Persist marca que el modelo recuerde la deteccion para hacer el track
        results = model.track(frame, classes=[0], persist=True)

        # plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
