import speech_recognition as sr
import os
import smtplib
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import lgpio
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import socket
import pickle
import google.generativeai as Genai
import os
import pyttsx3
import markdown2
from bs4 import BeautifulSoup
import threading
client_socket = None
server_socket = None

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

running = True


def object_detection():
    """Run object detection with distance sensing."""
    global running
    camera = setup_camera()
    previous_time = 0
    while running:
        frame = camera.capture_array()
        current_time = time.time()
        # Perform object detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result, objectInfo = getObjects(frame, 0.45, 0.2)
        
        # Measure distance
        distance = measure_distance()
        
        # Overlay distance information
        if distance is not None:
            cv2.putText(result, f"Distance: {distance} cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Alert if object is too close
            if distance < 50:
                if current_time - previous_time >= 2:
                    previous_time = current_time
                    cv2.putText(result, "WARNING: Object too close!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                           
                    speak("Warning: Object too close!")
        
        cv2.imshow("Object Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.05)
    
    cv2.destroyAllWindows()
    camera.close()

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

def send_audio_text(text):
    """Send text to be spoken on the Windows machine"""
    try:
        # Use the existing client socket
        data = {"type": "speak", "text": text}
        client_socket.send(pickle.dumps(data))
    except Exception as e:
        print(f"Error sending audio text: {e}")

def speak(text):
    """Send text to Windows for speech output"""
    send_audio_text(text)
    print(f"Speaking: {text}")

def listen_during_objectDetection():
    global running
    while running:
        command = listen()
        if "bye" or "goodbye" in command:
            running = False

def detect_object():
    listen_thread = threading.Thread(target=listen_during_objectDetection)
    listen_thread.daemon = True
    listen_thread.start()

    object_detection()
    
    global running
    running = False

    listen_thread.join(timeout = 1)


def listen():
    """Listen for commands from Windows client"""
    global client_socket
    try:
        data = client_socket.recv(4096)
        if data:
            text = pickle.loads(data)
            print(f"Received text: {text}")
            return text
    except Exception as e:
        print(f"Error receiving data: {e}")
        return ""
    return ""

def save_photo():
    camera = setup_camera()
    frame = camera.capture_array()
    camera.close()
    if frame is not None:
        photo_path = os.path.join(PHOTO_DIR, 'captured_photo.jpeg')
        cv2.imwrite(photo_path, frame)
        print("Photo captured and saved!")
        return True
    return False

def cleanup_resources():
    """Safely cleanup all resources"""
    global client_socket, server_socket
    
    # Close GPIO
    try:
        lgpio.gpiochip_close(h)
    except Exception as e:
        print(f"Error closing GPIO: {e}")
    
    # Close OpenCV windows
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error closing OpenCV windows: {e}")
    
    # Close sockets
    try:
        if client_socket is not None:
            client_socket.close()
    except Exception as e:
        print(f"Error closing client socket: {e}")
    
    try:
        if server_socket is not None:
            server_socket.close()
    except Exception as e:
        print(f"Error closing server socket: {e}")

def setup_socket_server(port=12345):
    """Setup socket server with proper error handling"""
    global client_socket, server_socket
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(1)
        print(f"Waiting for connection on port {port}...")
        client_socket, addr = server_socket.accept()
        print(f"Connected to {addr}")
        return True
    except Exception as e:
        print(f"Error setting up socket server: {e}")
        if server_socket:
            server_socket.close()
        return False

def ai():
    API_KEY = "AIzaSyDoAcbWpDqkMfmY-79CNzJBxZUQzgP3pQ0"
    Genai.configure(api_key=API_KEY)
    Genai.GenerationConfig(max_output_tokens=200)
    speak("Chat enabled")
    # Initialize co
    model = Genai.GenerativeModel()
    conversation = model.start_chat()
    while True:
    # Get user input as text through speech recognition
        user_input = listen().lower()
        if user_input is None:
          continue
        if("close chat" in user_input):
           speak("Chat closed.")
           return
        # Send user input to Gemini and get response
        response = conversation.send_message(user_input)
        # Assuming response is the GenerateContentResponse object
        if response:
        # Extract the generated content from the result
            generated_content = response.text
            html = markdown2.markdown(generated_content)
            soup = BeautifulSoup(html, 'html.parser')
            print(soup.get_text())
            speak(soup.get_text())
        else:
            speak("Connection not established.")
        

def main():
    global client_socket, server_socket
    try:
        if not setup_socket_server():
            print("Failed to setup socket server")
            return
            
        speak("Hello! How can I assist you today?")
        while True:
            query = listen().lower()
            
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
                detect_object()
            elif "open chat" in query:
                ai()
            elif "goodbye" in query or "good bye" in query:
                speak("Goodbye!")
                break
            else:
                print("Sorry, I didn't understand that.")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()

#audio bhi send karo
#check if it can run the model again or not
#change the color combination