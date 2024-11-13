import speech_recognition as sr
import pyttsx3
import os
import smtplib
import cv2
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from ultralytics import YOLO
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

model = tf.keras.models.load_model('model.h5')
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

def object_detection():
    try:
        import winsound
    except ImportError:
        print("Winsound Not found")

    # opening the file in read mode
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
    # Generate random colors for class list
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))

    # load a pretrained YOLOv8n model
    model = YOLO("weights/yolov8n.pt", "v8")

    # Vals to resize video frames | small frame optimise the run
    frame_wid = 640
    frame_hyt = 480

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("inference/videos/tikkinebanayi.mp4")

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # distance estimation function
    def distance_finder(focal_length, real_face_width, face_width_in_frame):
        distance = (real_face_width * focal_length) / face_width_in_frame
        return distance


    # face detector function
    def object_data(image,faces):

        object_width = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), WHITE, 1)
            face_width = w

        return face_width

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        i-=1
        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #  resize the frame | small frame optimise the run
        # frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Predict on image
        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        # Convert tensor array to numpy
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

        # Display the resulting frame
        cv2.imshow("ObjectDetection", frame)

        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# object distance model ------------------------------------------------>




# image captioning ---------------------------------------------------------->

def idx_to_word(integer, tokenizer):
    for word,index, in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      

def generate_caption(image_name):
    tokenizer = Tokenizer()
    image_id = image_name.split('.')[0]
    img_path = image_name
    image = Image.open(img_path)
#     captions = mapping[image_id]
#     print('---------------------Actual---------------------')
#     for caption in cnt(caaptions:
#         pription)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, 35)[8:][:-6]
    # print('--------------------Predicted--------------------')
    return y_pred



vocab = np.load(r'vocab.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}

recognizer = sr.Recognizer()
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

PHOTO_DIR = 'captured_photos'
os.makedirs(PHOTO_DIR, exist_ok=True)

def send_email(subject, body, to_email):
    smtp_server = 'smtp.gmail.com'  # Your SMTP server address
    smtp_port = 587  # Your SMTP server port (587 is the default for TLS)
    sender_email = 'hrishabhtiwari598@gmail.com'  # Your email address
    sender_password = 'olhd qnuw psdm hjqf'  # Your email password

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
    success, frame = cap.read()
    if success:
        photo_path = os.path.join(PHOTO_DIR, 'captured_photo.jpg')
        cv2.imwrite(photo_path, frame)

# model = load_model('my_model.h5')
# model.load_weights('mine_model_weights.h5')
# resnet = load_model('modele.h5')

# def image_captioning(image_path):
#     import cv2
#     import numpy as np
#     from keras.models import load_model
#     from keras.preprocessing.sequence import pad_sequences

#     embedding_size = 128
#     vocab_size = len(vocab)
#     max_len = 40

#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (224, 224))
#     image = np.reshape(image, (1, 224, 224, 3))

#     incept = resnet.predict(image).reshape(1, 2048)

#     text_in = ['startofseq']
#     final = ''

#     count = 0
#     while count < 20:
#         count += 1
#         encoded = []
#         for i in text_in:
#             encoded.append(vocab[i])

#         padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

#         sampled_index = np.argmax(model.predict([incept, padded]))

#         sampled_word = inv_vocab[sampled_index]

#         if sampled_word != 'endofseq':
#             final = final + ' ' + sampled_word
#         else:
#             break

#         text_in.append(sampled_word)

#     caption = final.strip()
#     return caption
#---------------------------------------------------------------------------------->

def main():
    greet()
    while True:
        query = listen().lower()
        print(query)
        if "hello" in query:
            speak("Hello, are you lost? Let me help you.")
        elif "what is your name" in query:
            speak("I'm a voice assistant, Sense it")
        elif "capture" in query:
            save_photo()
            speak("Photo captured and saved!")
            img_path = 'captured_photos\captured_photo.jpg'
            caption = image_captioning(img_path)
            print(caption)
            speak(caption)

        elif "help" in query:
            send_email("Help Needed", "I need assistance.", "kartik134yadav@gmail.com")
        elif "activate object" in query:
            object_detection()
        elif "goodbye" in query or "good bye" in query:
            speak("Goodbye!")
            break
        else:
            print("Sorry, I didn't understand that.")

if __name__ == "__main__":
    main()