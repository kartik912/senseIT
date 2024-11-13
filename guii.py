import streamlit as st
import speech_recognition as sr
import pyttsx3
import smtplib
import cv2  
from numpy import random
import winsound
import os
from ultralytics import YOLO
import folium

my_file = open("utils/coco.txt", "r")
    # reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()
WHITE = (255, 255, 255)
focal_length_found = 680.0
KNOWN_WIDTH = 8
# print(class_list)
fl={"person":550,"bicycle":800,"car":750,"traffic light":400,"fire hydrant":900,"stop sign":800,"backpack":860,"tie":680,"bottle":680,"wine glass":680,"cup":680,"chair":860,"laptop":800,"keyboard":800,"cell phone":680,"book":800,"clock":800}
aw={"person":40,"bicycle":100,"car":176,"traffic light":30,"fire hydrant":15,"stop sign":30,"backpack":35,"tie":8,"bottle":8,"wine glass":8,"cup":8,"chair":42,"laptop":30,"keyboard":30,"cell phone":8,"book":30,"clock":30}
#     
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
  engine.say(text)
  engine.runAndWait()


def createlocationFile():
    number = "+917668898255"
    from phonenumbers import geocoder
    import phonenumbers
    check_number = phonenumbers.parse(number)
    number_location = geocoder.description_for_number(check_number,"en")
    from phonenumbers import carrier
    service_provider = carrier.name_for_number(check_number,"en")
    from opencage.geocoder import OpenCageGeocode
    import geocoder
    g = geocoder.ip('me')
    lat,long = g.latlng
    from opencage.geocoder import OpenCageGeocode
    key = "0cf1ed0b5eb343a489284835405b0f65"
    geocoder = OpenCageGeocode(key)

    map_loaction = folium.Map(location = [lat,long],zoom_state=9)
    folium.Marker([lat,long],popup = check_number).add_to(map_loaction)
    map_loaction.save("mylocation.html")

def listen():
  with sr.Microphone() as source:
    print("Listening...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
  try:
    print("Recognizing...")
    query = recognizer.recognize_google(audio)
    return query.lower()
  except sr.UnknownValueError:
    print("Sorry, I didn't get that.")
    return ""

def greet():
  speak("Hello! How can I assist you today?")

def distance_finder(focal_length, real_face_width, face_width_in_frame):
        distance = (real_face_width * focal_length) / face_width_in_frame
        return distance

def object_data(image,faces):

        object_width = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), WHITE, 1)
            face_width = w

        return face_width

def capture():
    # Open the default camera (typically the first webcam connected)
    cap = cv2.VideoCapture(0)
    model = YOLO("weights/yolov8n.pt", "v8")
    # Check if the camera opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open camera.")
        return
    # Set up Streamlit elements
    frame_placeholder = st.empty()
    button_clicked1 = st.button("Close capture")
    # Read and display the webcam stream
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if the frame was captured successfully
        if not ret:
            st.error("Error: Unable to capture frame.")
            break
        
        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                
                text_bottom_right = (frame.shape[1] - 10, frame.shape[0] - 10)
                text_bottom_left = (10, frame.shape[0] - 10)
                font_scale = 1
                font_thickness = 2
                font = cv2.FONT_HERSHEY_COMPLEX
                # cv2.putText(frame, ("Distance" + str(Distance)), text_bottom_right, font, font_scale, (255, 255, 255), font_thickness)
                


                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                object_width_in_frame = object_data(frame,box.xywh.numpy())
                if object_width_in_frame != 0:
                    try:
                        Distance = distance_finder(fl[class_list[int(clsID)]],aw[class_list[int(clsID)]],object_width_in_frame)
                        if Distance < 50:
                            try:
                                winsound.Beep(500,1000)
                            except Exception as e:
                                os.system('beep -f %s -l %s' % (500,1000))
                    except Exception as e:
                        Distance = distance_finder(750,30,object_width_in_frame)

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + "Distance" + str(round(Distance,3)),
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
        # Display the frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        # Wait for the 'q' key to be pressed to quit
        if(button_clicked1):
           break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def send_email(subject, body, to_email):
    createlocationFile()
    import pywhatkit
    import smtplib
    from os.path import basename
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate
    smtp_server = 'smtp.gmail.com'  # Your SMTP server address
    smtp_port = 587  # Your SMTP server port (587 is the default for TLS)
    sender_email = 'hrishabhtiwari598@gmail.com'  # Your email address
    sender_password = 'olhd qnuw psdm hjqf'  # Your email password
    from datetime import datetime, timedelta

# Get the current time
    current_time = datetime.now()

    # Calculate the time after 1 minute
    one_minute_delta = timedelta(minutes=2)
    time_after_one_minute = current_time + one_minute_delta

    # Print the time after 1 minute in hour and minute
    time_after_one_minute_hour = time_after_one_minute.hour
    time_after_one_minute_minute = time_after_one_minute.minute

    message = MIMEMultipart()
    message['From']=sender_email
    message['To'] = COMMASPACE.join(to_email)
    message['Date'] = formatdate(localtime=True)
    message['Subject'] = subject
    message.attach(MIMEText(body))
    with open("mylocation.html","rb") as fil:
        part = MIMEApplication(fil.read(),Name=basename("mylocation.html"))
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename("mylocation.html")
        message.attach(part)
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())
        st.write("Email sent successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        server.quit()

    pywhatkit.sendwhatmsg("+919733253164","I am in danger please help check your mail for my location",time_after_one_minute_hour,time_after_one_minute_minute)
    send_email("HELP SOS","Plese help me i am in danger open this file in a browser to get my location","anusha572003@gmail.com")

def ai():
    import google.generativeai as Genai
    import os
    import pyttsx3
    import markdown2
    from bs4 import BeautifulSoup
    API_KEY = "AIzaSyBX1s2lkqipM_naWQU9HQhWtA2r7AY9i7E"
    Genai.configure(api_key=API_KEY)
    Genai.GenerationConfig(max_output_tokens=200)
    speak("Chat enabled")
    st.write("Chat enabled")
    button_clicked2 = st.button("close the chat")
    # Initialize co
    model = Genai.GenerativeModel()
    conversation = model.start_chat()
    while True:
    # Get user input as text through speech recognition
        user_input = listen()
        if user_input is None:
          continue
        if("good bye" in user_input or "goodbye" in user_input):
           speak("Good bye")
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
        if button_clicked2 :
            return
        

def image_captioning(image_path):
  # Implement image captioning logic here (you might need alternative libraries)
  st.write("Image captioning functionality not yet available.")

def main():
  st.title("Voice Assistant App")

  greet()

  while True:
    query = listen()
    st.write("You said:"+ query)

    if "hello" in query:
      speak("Hello, My name is SenseIt, Let me help you.")
    elif "what is your name" in query:
      speak("My name is SenseIt, I am a voice assistant. I am here to help.")
    elif "help" in query:
      send_email("Help Needed", "I need assistance.", "kartik134yadadv@gmail.com")
    elif "capture" in query:
       capture()
    elif "want to chat" in query:
       ai()
    elif "goodbye" in query or "good bye" in query:
       speak("It was nice meeting you, have a nice day!")
       break
    else:
      speak("Sorry, I didn't understand that.")

if __name__ == "__main__":
  main()
