import streamlit as st
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from keras.layers import Input, LSTM, Dense
from keras.models import Model

class ChatBot:
    def __init__(self):
        self.negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        self.exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
        self.training_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.input_docs = []
        self.target_docs = []
        self.input_tokens = set()
        self.target_tokens = set()
        self.input_features_dict = {}
        self.target_features_dict = {}
        self.reverse_input_features_dict = {}
        self.reverse_target_features_dict = {}
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.dimensionality = 256

    def load_model(self):
        self.training_model = load_model('training_model.h5')
        encoder_inputs = self.training_model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = self.training_model.layers[2].output
        self.encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

        latent_dim = self.dimensionality
        decoder_state_input_hidden = Input(shape=(latent_dim,))
        decoder_state_input_cell = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
        decoder_lstm = self.training_model.layers[3]
        decoder_inputs = self.training_model.input[1]
        decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_hidden, state_cell]
        decoder_dense = self.training_model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def decode_response(self, test_input):
        states_value = self.encoder_model.predict(test_input)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_features_dict['<START>']] = 1.
        decoded_sentence = ''
        stop_condition = False
        while not stop_condition:
            output_tokens, hidden_state, cell_state = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if (sampled_token == '<END>' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [hidden_state, cell_state]
        return decoded_sentence.replace("<START>", '').replace("<END>", '')

    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in self.input_features_dict:
                user_input_matrix[0, timestep, self.input_features_dict[token]] = 1.
        return user_input_matrix

    def make_exit(self, reply):
        return any(cmd in reply.lower() for cmd in self.exit_commands)

def main():
    st.title("ChatBot")
    st.write("Welcome to the ChatBot! Type your message below.")

    chatbot = ChatBot()
    chatbot.load_model()

    # Load dataset
    data = pd.read_csv("mentalhealth.csv")
    chatbot.input_docs = data['Questions'].tolist()
    chatbot.target_docs = data['Answers'].tolist()
    chatbot.preprocess_data()

    user_input = st.text_input("You:", "")
    if user_input:
        if chatbot.make_exit(user_input):
            st.write("ChatBot: Ok, have a great day!")
        else:
            input_matrix = chatbot.string_to_matrix(user_input)
            response = chatbot.decode_response(input_matrix)
            st.write("ChatBot:", response)

if __name__ == "__main__":
    main()
