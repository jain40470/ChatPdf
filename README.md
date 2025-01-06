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
