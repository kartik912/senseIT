import speech_recognition as sr
import pyttsx3
import os
import smtplib
import cv2
import numpy as np
import time
print("Imported all libraries")
from picamera2 import Picamera2
import lgpio
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# GPIO Setup for Ultrasonic Sensor
TRIG_PIN = 4
ECHO_PIN = 17
CHIP = 0

# Initialize GPIO
h = lgpio.gpiochip_open(CHIP)
lgpio.gpio_claim_output(h, TRIG_PIN)
lgpio.gpio_claim_input(h, ECHO_PIN)

# Load the class names
classNames = []
classFile = "/home/kartik/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load DNN model
configPath = "/home/kartik/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/kartik/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize image captioning models
model2 = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def measure_distance():
    """Measure distance using ultrasonic sensor with lgpio."""
    lgpio.gpio_write(h, TRIG_PIN, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, TRIG_PIN, 0)

    start_time = time.time()
    timeout = start_time + 1

    while lgpio.gpio_read(h, ECHO_PIN) == 0:
        start_time = time.time()
        if time.time() > timeout:
            return None

    while lgpio.gpio_read(h, ECHO_PIN) == 1:
        stop_time = time.time()
        if time.time() > timeout:
            return None

    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2

    return round(distance, 1)

def setup_camera():
    """Initialize and configure the Raspberry Pi camera."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 320)})
    picam2.configure(config)
    picam2.start()
    return picam2

def getObjects(img, thres, nms, draw=True, objects=[]):
    """Detect objects in an image using the pre-trained DNN model."""
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)),
                              (box[0] + 200, box[1] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, objectInfo

def predict_step(image_paths):
    """Generate image captions using the pre-trained model."""
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model2.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def object_detection():
    """Run object detection with distance sensing."""
    camera = setup_camera()
    
    while True:
        frame = camera.capture_array()
        
        # Perform object detection
        result, objectInfo = getObjects(frame, 0.45, 0.2)
        
        # Measure distance
        distance = measure_distance()
        
        # Overlay distance information
        if distance is not None:
            cv2.putText(result, f"Distance: {distance} cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Alert if object is too close
            if distance < 50:
                cv2.putText(result, "WARNING: Object too close!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Object Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.05)
    
    cv2.destroyAllWindows()

# Speech recognition and other helper functions
recognizer = sr.Recognizer()
engine = pyttsx3.init()

PHOTO_DIR = 'captured_photos'
os.makedirs(PHOTO_DIR, exist_ok=True)

def send_email(subject, body, to_email):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = 'hrishabhtiwari598@gmail.com'
    sender_password = 'olhd qnuw psdm hjqf'

    message = f'Subject: {subject}\n\n{body}'

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.quit()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        return query
    except sr.UnknownValueError:
        print("Sorry, I didn't get that.")
        return ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

def greet():
    speak("Hello! How can I assist you today?")

def save_photo():
    camera = setup_camera()
    frame = camera.capture_array()
    if frame is not None:
        photo_path = os.path.join(PHOTO_DIR, 'captured_photo.jpeg')
        cv2.imwrite(photo_path, frame)
        print("Photo captured and saved!")
        return True
    return False

def main():
    try:
        greet()
        while True:
            query = listen().lower()
            print(query)
            
            if "hello" in query:
                speak("Hello, are you lost? Let me help you.")
            elif "what is your name" in query:
                speak("I'm a voice assistant, Sense it")
            elif "capture" in query:
                if save_photo():
                    speak("Photo captured and saved!")
                    caption = predict_step(['captured_photos/captured_photo.jpeg'])
                    print(caption)
                    speak(caption[0] if caption else "No caption generated")
            elif "help" in query:
                send_email("Help Needed", "I need assistance.", "kartik134yadav@gmail.com")
            elif "activate object" in query:
                speak("Activating object detection")
                object_detection()
            elif "goodbye" in query or "good bye" in query:
                speak("Goodbye!")
                break
            else:
                print("Sorry, I didn't understand that.")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        lgpio.gpiochip_close(h)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()