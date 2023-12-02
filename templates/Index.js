const videoCanvas = document.getElementById('camera-feed');
const videoContext = videoCanvas.getContext('2d');
const captureButton = document.getElementById('capture-button');
const responseBox = document.getElementById('response-box');
const serverResponseImage = document.getElementById('server-response-image');
let videoStream;

// Access the camera and display the video feed
async function startCamera() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        const facingMode = 'environment'; // Set to 'environment' for the back camera

        const constraints = {
            video: { facingMode },
        };

        videoStream = await navigator.mediaDevices.getUserMedia(constraints);

        const videoElement = document.createElement('video');
        videoElement.srcObject = videoStream;
        videoElement.addEventListener('loadedmetadata', () => {
            const { videoWidth, videoHeight } = videoElement;
            videoCanvas.width = videoWidth;
            videoCanvas.height = videoHeight;
            drawVideoFrame(videoElement, videoContext);
        });
        videoElement.play();
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Function to draw the video frame on the canvas
function drawVideoFrame(video, context) {
    context.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);

    // Draw a square on the canvas
    const squareSizePixels = 210;
    const squareColor = 'green';
    const x = (videoCanvas.width - squareSizePixels) / 2;
    const y = (videoCanvas.height - squareSizePixels) / 2;

    context.strokeStyle = squareColor;
    context.lineWidth = 2;
    context.strokeRect(x, y, squareSizePixels, squareSizePixels);

    requestAnimationFrame(() => drawVideoFrame(video, context));
}

// Function to capture a photo and send it to the server
async function captureAndSend() {
    if (videoStream) {
        // Pause the video feed
        videoStream.getTracks().forEach(track => track.stop());


        // Capture the current frame from the video feed
        const capturedFrame = document.createElement('canvas');
        capturedFrame.width = videoCanvas.width;
        capturedFrame.height = videoCanvas.height;
        const capturedFrameContext = capturedFrame.getContext('2d');
        capturedFrameContext.drawImage(videoCanvas, 0, 0, videoCanvas.width, videoCanvas.height);

        // Convert the captured frame to a Blob with JPG format
        const blob = await new Promise(resolve => capturedFrame.toBlob(resolve, 'image/jpeg'));

        // Display the captured photo in the response box
        serverResponseImage.src = URL.createObjectURL(blob);

        // Send the image to the server
        const formData = new FormData();
        formData.append('file', blob, 'captured_photo.jpg');

        try {
            const response = await fetch('https://436d-94-139-191-146.ngrok-free.app/get_image', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                // Handle the response from the server
                const processedBlob = await response.blob();
                serverResponseImage.src = URL.createObjectURL(processedBlob);
                responseBox.style.display = 'block';
            } else {
                const errorResult = await response.text();
                alert(errorResult);
            }
        } catch (error) {
            alert('Error sending image to the server:', error);
        }
    }
}

// Start the camera when the page loads
document.addEventListener('DOMContentLoaded', startCamera);
