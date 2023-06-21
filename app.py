# Import the required modules
from flask import Flask, render_template, Response
import cv2
import numpy as np

# Create the Flask application
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Generator function for streaming video frames
def stream_frames():
    cap = cv2.VideoCapture(1)

    # Set the frame width and height
    framewidth = 640
    frameheight = 480

    # Set the camera properties
    cap.set(3, framewidth)
    cap.set(4, frameheight)

    # Define a color dictionary to map RGB values to color names
    color_dict = {
        (200, 160, 180): "Fresh",
        (170, 170, 180): "Still Fresh",
        (100, 200, 210): "Past Best",
    }

    # Define a function to get the closest color in the color dictionary
    def get_closest_color(color):
        # Calculate the distance between the color and each color in the dictionary
        distances = {np.linalg.norm(np.array(color) - np.array(k)): v for k, v in color_dict.items()}

        # Return the color name with the minimum distance
        return distances[min(distances)]

    # Initialize variables for circle detection
    prevcircle = None
    dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale frame
        blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)

        # Detect circles in the blurred frame
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                                   param1=100, param2=40, minRadius=20, maxRadius=150)

        # Check if circles are detected
        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None

            # Find the closest circle to the previous circle
            for i in circles[0, :]:
                if chosen is None:
                    chosen = i
                if prevcircle is not None:
                    if dist(chosen[0], chosen[1], prevcircle[0], prevcircle[1]) <= dist(i[0], i[1], prevcircle[0], prevcircle[1]):
                        chosen = i

            # Update the previous circle
            prevcircle = chosen

            # Extract the region of interest (circle) from the frame
            x, y, radius = chosen
            roi = frame[y - radius: y + radius, x - radius: x + radius]

            # Get the average color in the circular region
            average_color = cv2.mean(roi)[:3]

            # Get the closest color name from the color dictionary
            color_name = get_closest_color(average_color)

            # Draw the detected circle and display the color name
            cv2.circle(frame, (x, y), radius, (255, 0, 255), 3)
            cv2.putText(frame, f"Color: {color_name}", (int(x - radius), int(y - radius - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Convert the OpenCV frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        # Yield the frame data as a response to update the web page
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    # Release the camera
    cap.release()

# Route for streaming the video feed
@app.route('/video_feed')
def video_feed():
    return Response(stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
