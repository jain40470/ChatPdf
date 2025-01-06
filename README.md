# Chat PDF Application

This application allows users to upload PDF files, extract the text content, and interact with it using a question-answering system powered by a language model.
The PDF content is processed into chunks, vectorized for semantic search, and users can ask questions related to the text. 
The application provides a detailed response with relevant portions of the pdf.

## Features

- **PDF Upload:** Upload one or more PDF files.
- **Text Extraction:** Extracts and displays the content of uploaded PDFs.
- **Chunking:** Splits large texts into manageable chunks for efficient processing.
- **Vectorization:** Converts the text into embeddings for semantic search.
- **Question-Answering:** Ask questions and get relevant answers from the PDF content, with  relevant portions of the pdf.

## Technologies Used

- **Streamlit:** For building the web interface.
- **PyPDF2:** For PDF text extraction.
- **Langchain:** For handling text chunking, embeddings, and question answering.
- **Google Gemini API:** For generating embeddings and chat-based responses.
- **FAISS:** For fast similarity search of vector embeddings.
- **dotenv:** For loading environment variables, such as API keys.

## Setup and Installation

1. Clone this repository to your local machine.
2. Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - For macOS/Linux:

      ```bash
      source venv/bin/activate
      ```

    - For Windows:

      ```bash
      .\venv\Scripts\activate
      ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Set up your Google API key:
   - Create a `.env` file in the root directory and add your Google API key as follows:

    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

6. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Deployment

The application is also deployed on Streamlit Cloud and can be accessed at:  
**[chatpdfyup.streamlit.app]((https://chatpdfyup.streamlit.app))**

## Usage

1. Upload your PDF files through the file uploader.
2. After processing, you can ask questions about the content of the PDF.
3. The system will provide an answer and relevant portions of the pdf.

## Structure

- `app.py`: Main Streamlit application.
- `requirements.txt`: List of dependencies.
- `.env`: Environment file to store sensitive API keys.

## FAQ

### What is the chunk size for the PDFs?
By default, the application splits the text into chunks of 5000 characters with an overlap of 1000 characters.

### How does the question-answering work?
The system converts both the user's question and the PDF chunks into embeddings. The embeddings are then compared to find the most relevant chunk based on semantic similarity using FAISS. The matched context is used to generate a detailed response.

### Can I upload multiple PDFs at once?
Yes, you can upload multiple PDF files at once, and the application will process them sequentially.

### What happens if my PDF is too large?
If your PDF is too large, the application splits the content into smaller chunks (5000 characters) to efficiently process it for embedding and semantic search.

### Can I highlight multiple portions of the PDF in the response?
Currently, the application provides only the first relevant portion of the text that matches the user's question. 

### How accurate are the answers provided by the application?
The accuracy of the answers depends on the quality of the PDF content and the relevance of the chunks to the user’s question. The application uses semantic search to identify the most relevant chunk based on vector embeddings.

### Can I use a custom language model for embeddings and answering?
Yes, the application uses Google Gemini for embeddings and answering, but you can switch to other models like OpenAI's GPT or Hugging Face's models by modifying the code.

### What is FAISS and why is it used in this application?
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. It is used in this application to find the most similar chunk from the vector embeddings based on the user’s query.

### What is the role of the Google API key?
The Google API key is required to use the Google Gemini API for generating embeddings and answering questions. You can get the API key from the Google Cloud Console.

### Can I run this application offline?
No, the application relies on the Google Gemini API and FAISS for embeddings and search, which requires an internet connection.

### Is this project open source?
Yes, this project is open source. Feel free to contribute and suggest improvements.

### Can I customize the chunk size or overlap?
Yes, you can adjust the chunk size and overlap by modifying the `RecursiveCharacterTextSplitter` parameters in the code. 

### How can I deploy this application on my own server?
You can deploy this application on any platform that supports Python and Streamlit, such as Heroku, AWS, or your local server. Follow the setup instructions in the README for deployment.

### How do I contact the developer for support?
You can contact the developer by opening an issue in the repository or reaching out via email (jain40470@gmail.com)


