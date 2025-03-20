# Paper Bold AI

## Project Overview

Paper Bold AI is an AI-powered web application designed for academic paper analysis. Built with a Retrieval Augmented Generation (RAG) architecture, it helps researchers and students efficiently extract key insights from complex academic papers in PDF format.

The application addresses the time-consuming challenge of understanding lengthy research papers by providing instant summaries, identifying technical approaches, and enabling conversational Q&A about the paper's content.

## RAG Architecture

Paper Bold AI implements a complete RAG pipeline:

1. **Content Extraction**: Converts PDF documents to text using PyPDF2
2. **Intelligent Chunking**: Dynamically segments text based on document size and structure
3. **Vector Embedding**: Transforms text chunks into semantic vectors using Google's embedding model
4. **Vector Storage**: Organizes vectors in a Chroma database for quick retrieval
5. **Context Matching**: ConversationalRetrievalChain query processing
6. **Response Generation**: Gemini-powered answer synthesis

## Deploy Link

[https://paper-bold.onrender.com/](https://paper-bold.onrender.com/)


<img src="https://github.com/user-attachments/assets/b08ab000-a645-43cd-b53f-c441119cf874" width="800">


## Dataset

This application processes PDF files uploaded by users. No pre-prepared dataset is used. Each PDF is processed after upload and converted into a vector database.

## Features and Use Cases

- **Bilingual Interface**: Full support for both Turkish and English throughout the application
- **Adaptive Document Processing**: Automatically adjusts chunking parameters based on PDF size for optimal performance
- **AI-Powered Summarization**: Generates concise, focused summaries highlighting key findings and methodologies
- **Technical Element Detection**: Automatically identifies and extracts models, algorithms, and technical approaches mentioned in papers
- **Contextual Question Answering**: Allows natural language queries about any aspect of the paper with accurate, citation-based responses
- **Responsive Design**: Fully functional across desktop and mobile devices

**Ideal For:**
- Academic researchers conducting literature reviews
- Students navigating complex research papers
- Research teams sharing and discussing technical publications
- Anyone needing to quickly extract specific information from academic papers

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **GenAI**: Gemini 1.5 Flash, Google Embedding, LangChain
- **Vector Database**: Chroma
- **PDF Processing**: PyPDF2

## Local Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/enesmanan/paper-bold.git
   cd paper-bold
   ```

2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Go to `http://localhost:5000` in your browser

## Contact

Please get in touch if you have any questions about the project.

- **E-mail:** [enesmanan768@gmail.com](mailto:enesmanan768@gmail.com)
- **GitHub:** [github.com/enesmanan](https://github.com/enesmanan)
- **LinkedIn:** [linkedin.com/in/enesfehmimanan](https://linkedin.com/in/enesfehmimanan)

## Repository Structure

```
paper-bold/
├── app.py                  
├── static/                 
│   ├── style.css           
│   └── images/             
├── templates/              
│   ├── index.html          
│   └── viewer.html         
├── uploads/                
├── chroma_db/              
├── requirements.txt       
└── README.md              
```
