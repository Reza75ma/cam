# Import the required modules
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

# Create the Flask application
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Process the frame and detect circles
def process_frame(frame):
    # Convert the frame to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)

    # Detect circles in the blurred frame
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=40, minRadius=20, maxRadius=150)

    # Process the detected circles and colors
    processed_circles = []
    colors = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            # Extract the circle coordinates and radius
            x, y, radius = circle

            # Extract the region of interest (circle) from the frame
            roi = frame[y - radius: y + radius, x - radius: x + radius]

            # Get the average color in the circular region
            average_color = cv2.mean(roi)[:3]

            # Append the circle and color information to the result lists
            processed_circles.append((x, y, radius))
            colors.append(average_color)

    return processed_circles, colors

# Generator function for streaming video frames
def stream_frames():
    cap = cv2.VideoCapture(0)

    # Set the frame width and height
    framewidth = 640
    frameheight = 480

    # Set the camera properties
    cap.set(3, framewidth)
    cap.set(4, frameheight)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame and detect circles
        circles, colors = process_frame(frame)

        # Draw the detected circles on the frame
        for circle in circles:
            x, y, radius = circle
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)

        # Convert the OpenCV frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        # Send the detected circles and colors as JSON response
        response_data = {'circles': circles, 'colors': colors}
        yield jsonify(response_data)

    # Release the camera
    cap.release()

# Route for processing the video frame
@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    frame_data = request.form['frame_data']
    nparr = np.fromstring(frame_data.decode('base64'), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    circles, colors = process_frame(frame)

    response_data = {'circles': circles, 'colors': colors}
    return jsonify(response_data)

# Route for streaming the video feed
@app.route('/video_feed')
def video_feed():
    return Response(stream_frames(), mimetype='application/json')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
