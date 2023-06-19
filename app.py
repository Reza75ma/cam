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

    color_dict = {
        (200, 160, 180): "Fresh",
        (170, 170, 180): "Still Fresh",
        (100, 200, 210): "Past Best",
    }

    def get_closest_color(color):
        distances = {np.linalg.norm(np.array(color) - np.array(k)): v for k, v in color_dict.items()}
        return distances[min(distances)]

    prevcircle = None
    dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                                   param1=100, param2=40, minRadius=20, maxRadius=150)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None

            for i in circles[0, :]:
                if chosen is None:
                    chosen = i
                if prevcircle is not None:
                    if dist(chosen[0], chosen[1], prevcircle[0], prevcircle[1]) <= dist(i[0], i[1], prevcircle[0], prevcircle[1]):
                        chosen = i

            prevcircle = chosen
            x, y, radius = chosen
            roi = frame[y - radius: y + radius, x - radius: x + radius]
            average_color = cv2.mean(roi)[:3]
            color_name = get_closest_color(average_color)

            cv2.circle(frame, (x, y), radius, (255, 0, 255), 3)
            cv2.putText(frame, f"Color: {color_name}", (int(x - radius), int(y - radius - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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
