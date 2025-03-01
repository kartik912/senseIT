import google.generativeai as Genai
import os
import pyttsx3
import markdown2
from bs4 import BeautifulSoup

def ai():
    API_KEY = "AIzaSyDoAcbWpDqkMfmY-79CNzJBxZUQzgP3pQ0"
    Genai.configure(api_key=API_KEY)
    Genai.GenerationConfig(max_output_tokens=200)
    # speak("Chat enabled")
    print("hn bol")
    # st.write("Chat enabled")
    # button_clicked2 = st.button("close the chat")

    # Initialize co
    model = Genai.GenerativeModel()
    conversation = model.start_chat()
    while True:
    # Get user input as text through speech recognition
        # user_input = listen()
        user_input = input("HN bol : ")
        if user_input is None:
          continue
        if("good bye" in user_input or "goodbye" in user_input):
        #    speak("Good bye")
           print("Good bye")
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
            # speak(soup.get_text())
        else:
            # speak("Connection not established.")
            print("Abe sale connect nhi hora")
        # if button_clicked2 :
        #     return
        
ai()