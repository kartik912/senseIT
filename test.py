
import google.generativeai as Genai
import os
import pyttsx3
import markdown2
from bs4 import BeautifulSoup
API_KEY = "AIzaSyBX1s2lkqipM_naWQU9HQhWtA2r7AY9i7E"
Genai.configure(api_key=API_KEY)
Genai.GenerationConfig(max_output_tokens=200)
# Initialize co
model = Genai.GenerativeModel()
# Function to convert text to speech (using Python's built-in library)
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

  # Play the audio file (implementation depends on your OS)
  # ...

# Function to recognize speech (using SpeechRecognition library - install separately)
def listen():
  import speech_recognition as sr
  r = sr.Recognizer()
  with sr.Microphone() as source:
    print("Listening...")
    audio = r.listen(source)
  try:
    text = r.recognize_google(audio)
    return text.lower()
  except sr.UnknownValueError:
    print("Sorry, could not understand audio")
    return None

def main():
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

main()