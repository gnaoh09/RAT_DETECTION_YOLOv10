import time
import cv2
import onnxruntime as ort
import numpy as np

# Initialize the camera
camera = PiCamera()
camera.resolution = (224, 224)  # Adjust resolution as needed
rawCapture = PiRGBArray(camera, size=(224, 224))

# Allow the camera to warm up
time.sleep(0.1)

# Load the ONNX model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Preprocess the image (adjust according to your model requirements)
    input_frame = image.transpose(2, 0, 1)  # Change data layout from HWC to CHW
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {input_name: input_frame})

    # Post-process and visualize the results (customize based on your model's output)
    # For example, if detecting objects:
    # bounding_boxes, scores, class_ids = process_output(outputs)
    # visualize_detections(image, bounding_boxes, scores, class_ids)

    # Display the resulting frame
    cv2.imshow("Frame", image)
    
    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close windows
cv2.destroyAllWindows()
