# AI-Based Chatbot

An AI-powered chatbot capable of understanding and responding to user queries using natural language processing (NLP), OpenAI GPT models, and speech recognition.

---

## **Features**
- NLP-based sentence similarity evaluation for intelligent query handling.
- Integration with OpenAI API for generating dynamic and accurate responses.
- Speech-to-text functionality using Google Speech Recognition API.
- Duplicate query detection using cosine similarity.
- Interactive voice-based user interface.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Jeevika-P/AI-Based-ChatBot
   cd ai-chatbot

## **Project Structure**
ai-chatbot/
│
├── data/
│   └── quora_duplicate_train_small.zip  # Example dataset
│
├── chatbot.py                           # Main chatbot script
├── requirements.txt                     # Dependencies
├── README.md                            # Project documentation


**How It Works**
1.Sentence Similarity Evaluation:
    Uses cosine similarity to detect duplicate queries.
    Processes a dataset of question pairs to train the model.
2.Dynamic Responses:
    OpenAI GPT API generates meaningful responses to user queries.
3.Speech Recognition:
     Captures and processes user input through voice commands.

## **Dependencies**
- `pandas`
- `numpy`
- `scikit-learn`
- `speechrecognition`
- `pyttsx3`
- `openai`

## **Future Enhancements**
- Add multi-language support for a wider user base.
- Integrate sentiment analysis to improve response quality.
- Enhance duplicate detection accuracy with advanced embeddings.

## **Contributing**
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## **Author**
- **Jeevika P**  
- Connect with me on [LinkedIn](https://www.linkedin.com/in/jeevika2455/)


