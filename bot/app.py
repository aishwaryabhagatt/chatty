import streamlit as st
import pandas as pd

class ChatBot:
    def __init__(self, dataset_path):
        try:
            self.dataset = pd.read_csv(dataset_path)
        except FileNotFoundError:
            st.error(f"File '{dataset_path}' not found. Please make sure the file exists.")
            st.stop()

    def generate_response(self, user_input):
        user_input = user_input.lower()
        response = "I'm sorry, I'm not sure how to respond to that."
        for index, row in self.dataset.iterrows():
            if row['Questions'].lower() in user_input:
                response = row['Answers']
                break
        return response

def main():
    st.title("ChatBot")
    st.write("Welcome to the ChatBot! Type your message below.")

    chatbot = ChatBot("mentalhealth.csv")

    user_input = st.text_input("You:", "")
    if user_input:
        response = chatbot.generate_response(user_input)
        st.write("ChatBot:", response)

if __name__ == "__main__":
    main()
