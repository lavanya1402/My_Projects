RAG Chatbot Project

This is a **Retrieval-Augmented Generation (RAG)** chatbot that provides intelligent responses based on the content of uploaded documents. The system allows users to upload a **PDF** or **text file**, and then interact with the chatbot to ask questions based on the document's content.

## Features

- **Document Upload**: Users can upload a PDF or text file that contains information they want to query about.
- **Intelligent Responses**: The chatbot retrieves relevant data from the document and generates accurate responses based on the input.
- **Customizable**: The bot can be easily trained with different documents, making it adaptable to various contexts.

## Technologies Used

- **Streamlit**: For building the user interface.
- **LangChain**: Used for integrating language models and chaining multiple components.
- **FAISS**: A library for efficient similarity search and clustering.
- **Sentence-Transformers**: For encoding text into vector representations.
- **Groq**: A framework used for accelerating AI models.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lavanya1402/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Interact with the Chatbot**:
   Once the app is running, open the provided URL in a browser. You can then upload your document (PDF or text file) and start interacting with the chatbot.

## Contribution


This README will give a user a clear overview of what your project does, how to install it, and what technologies are used.
