import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
from vectorizers.factory import get_vectoriser
import openai
import speech_recognition as sr
import pyttsx3


class SentenceSimilarityEvaluation:
    def __init__(self, zipfilename, vectorizer_type):
        self.df = None
        self.read_data(zipfilename)
        all_unique_questions = self.get_corpus()
        self.build_model(vectorizer_type, all_unique_questions)

    def get_corpus(self):
        q1_column = self.df["question1"].tolist()
        q2_column = self.df["question2"].tolist()
        unique_qs = set(q1_column + q2_column)
        return list(unique_qs)

    def read_data(self, zipfilename):
        with zipfile.ZipFile(zipfilename) as z:
            csvfilename = zipfilename.replace(".zip", ".csv").replace("data/", "")
            print(csvfilename)
            with z.open(csvfilename) as f:
                self.df = pd.read_csv(f)
                self.df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)
                self.df.dropna(inplace=True)

    def build_model(self, vectorizer_type, questions):
        self.vectorizer = get_vectoriser(vectorizer_type)
        self.vectorizer.vectorize(questions)

    def check_duplicate(self):
        computed_is_duplicate = []
        n_matching_rows = 0

        for index, row in self.df.iterrows():
            if index >= 10:  # Limiting to a small sample
                break

            q1 = row['question1']
            q2 = row['question2']
            is_duplicate = row['is_duplicate']

            q1_array = self.vectorizer.query(q1)
            q2_array = self.vectorizer.query(q2)

            sims = cosine_similarity(q1_array, q2_array)

            c_is_duplicate = 1 if sims[0][0] > 0.9 else 0
            computed_is_duplicate.append(c_is_duplicate)

            if c_is_duplicate == is_duplicate:
                n_matching_rows += 1

        accuracy = n_matching_rows / len(computed_is_duplicate)
        return accuracy


# OpenAI Integration
def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return "Error recognizing audio"


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]


def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        print("Say 'Genius' to start recording your question")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "genius":
                    print("Say your question")
                    with sr.Microphone() as source2:
                        audio = recognizer.listen(source2, phrase_time_limit=None, timeout=None)
                    with open("input.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    
                    text = transcribe_audio_to_text("input.wav")
                    if text:
                        print(f"You said: {text}")
                        response = generate_response(text)
                        print(f"ChatGPT says: {response}")
                        speak_text(response)
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Initialize OpenAI API Key
    openai.api_key = "your-openai-api-key"

    # Test the SentenceSimilarityEvaluation
    zipcsvfile = "data/quora_duplicate_train_small.zip"
    senteval = SentenceSimilarityEvaluation(zipcsvfile, 'spacy')
    accuracy = senteval.check_duplicate()
    print(f"Duplicate Detection Accuracy: {accuracy}")

    # Run the chatbot
    main()
