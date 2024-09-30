# DocuChat AI: LangChain-powered Gemini Chatbot

DocuChat AI is an intelligent document-based chatbot that leverages the power of Google's Gemini AI and LangChain to provide accurate and context-aware responses. This application allows users to upload various document types and engage in conversations based on the content of those documents.

## Features

- Document upload support for PDF, DOCX, Excel, JSON, and TXT files
- Intelligent text extraction and processing
- Advanced context retrieval using LangChain and FAISS
- Conversational AI powered by Google's Gemini and PaLM models
- User-friendly interface built with Streamlit

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/docuchat-ai.git
   cd docuchat-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Obtain API keys for Google Gemini and Google PaLM
   - Set them as environment variables or update the placeholders in the code

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

3. Use the sidebar to upload documents

4. Start chatting with the AI about the content of your documents!

## How It Works

1. **Document Processing**: When a document is uploaded, the application extracts text and splits it into manageable chunks.

2. **Embedding and Indexing**: The text chunks are embedded using HuggingFace's sentence transformers and indexed in a FAISS vector store for efficient retrieval.

3. **Query Processing**: When a user asks a question, the application uses LangChain's ConversationalRetrievalChain to:
   - Retrieve relevant context from the indexed documents
   - Generate a response using the Gemini model, taking into account the retrieved context and conversation history

4. **Response Generation**: The AI-generated response is displayed to the user, maintaining a conversational flow.

## Contributing

Contributions to DocuChat AI are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini and PaLM for providing powerful language models
- LangChain for the excellent tools and abstractions for building LLM-powered applications
- Streamlit for the intuitive app framework
