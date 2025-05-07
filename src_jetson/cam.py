import nanocamera as nano
import cv2

# Create the camera instance (USB mode)
camera = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30)

# Check if camera is ready
if not camera.isReady():
    print("Cannot open NanoCamera")
    exit()

# Your text detection function
def detect_text(frame):
    """
    Replace this with your actual text detection model.
    For demo purposes, this just returns the same frame.
    """
    # Example: run your text detection here
    # result = model_inference(frame)

    # For demo: draw dummy text
    cv2.putText(frame, "Text detection running", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

while True:
    # Read a frame from the camera
    frame = camera.read()
    if frame is None:
        print("Failed to capture image")
        break

    # Run text detection
    output_frame = detect_text(frame)

    # Display the frame
    cv2.imshow("NanoCamera Text Detection", output_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and windows
camera.release()
cv2.destroyAllWindows()

