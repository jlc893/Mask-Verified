from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model(os.path.join('AI','masktensor'))

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    # Get your video capture
    cap = cv2.VideoCapture(0)

    # Create haarcascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    color = (255, 0, 0)
    #Refresh html page; continuously send a new image to be displayed through each loop
    while True:
        # maskcheck is True if a face is detected
        maskcheck = False
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        # Error check; break out of while if camera does not respond properly
        if not success:
            print("not success")
            break
        # If the capture is successful, display it
        else:
            # Convert images to gray scale so it can be used by eye haarcascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in image. Returns a list of tuples containing the eye coordinates
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

            # Make a box around a person's head based on the distance between the centers of their eyes
            coord = []
            x0 = 0
            y0 = 0
            # If the eyes are detected
            if len(eyes) == 2:
                # Find the centers of the eyes
                for (cx, cy, cw, ch) in eyes:
                    coord.append((cx + cw / 2, cy + ch / 2))
                # Distance between the eyes
                dist = ((coord[1][0] - coord[0][0]) ** 2 + (coord[1][1] - coord[0][1]) ** 2) ** .5
                # Center of the face
                center = ((coord[0][0] + coord[1][0]) / 2, (coord[0][1] + coord[1][1]) / 2)
                # half of the side of the box being created
                radius = 2 * dist
                x0 = round(center[0] - radius)
                y0 = round(center[1] - radius)
                x1 = round(center[0] + radius)
                y1 = round(center[1] + radius)
                # Make the rectangle
                rect = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                # Save the rectangle as an array
                roi_color = frame[y0:y1, x0:x1]
                # Check if the person is wearing a mask
                maskcheck = True

            # If color is green display "Thanks for wearing your mask" text
            if color == (0, 255, 0):
                cv2.putText(rect, f'Thanks for wearing your mask : {str(round(prediction[0][1], 3))}', (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # If color is red display "Please put on a mask" text
            elif color == (0, 0, 255):
                cv2.putText(rect, f'Please put on a mask : {str(round(prediction[0][1], 3))}', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display image
            cv2.imshow('frame', frame)

            # If the key "q" is pressed the program will quit
            if cv2.waitKey(1) == ord('q'):
                break

            # Return the frame (the captured image) with all edits
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            if maskcheck == True:
                # Checks if the person is wearing a mask or not using our trained Tensorflow AI model
                array = np.asarray(cv2.resize(roi_color, (30, 30), interpolation=cv2.INTER_AREA))
                input_array = np.expand_dims(array, axis=0)
                prediction = model.predict(input_array)

                # Determine whether the mask is on or not based on the weighted returned values
                if prediction[0][1] > .99:
                    color = (0, 255, 0)
                    cv2.putText(rect, f'Thanks for wearing your mask : {prediction[0][1]}', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    color = (0, 0, 255)
                    cv2.putText(rect, f'Please put on a mask : {prediction[0][1]}', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)




@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    #Video streaming homepage
    return render_template('index.html')

def maskAI(img):
    return model.predict(img)

if __name__ == '__main__':
    app.run(debug=True)