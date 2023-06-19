from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def stream_frames():
    cap = cv2.VideoCapture(0)
    framewidth = 640
    frameheight = 480
    cap.set(3, framewidth)
    cap.set(4, frameheight)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)

        circles = cv2.HoughCircles(
            blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=40, minRadius=20, maxRadius=150
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                x, y, radius = circle
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
