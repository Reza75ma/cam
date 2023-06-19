from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Route for processing the video frame and detecting circles
@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.form['frame_data']

    # Decode the frame data from base64
    frame_decoded = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(frame_decoded, cv2.IMREAD_COLOR)

    # Process the frame, detect circles, and get the colors
    # Replace this with your circle detection and color extraction logic
    circles = []
    colors = []

    # Example code to detect circles (replace with your implementation)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=40, minRadius=20, maxRadius=150)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            x, y, radius = circle
            roi = frame[y - radius: y + radius, x - radius: x + radius]
            average_color = cv2.mean(roi)[:3]
            colors.append(average_color)

    # Prepare the response data
    response_data = {
        'circles': circles.tolist(),
        'colors': colors
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
