import os
import shutil

import google.generativeai as genai
from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template, request, send_file, session)
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.globals import set_verbose
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from PyPDF2 import PdfReader

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CHROMA_DIR = "chroma_db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "paper_bold_secret_key"  # Required for session

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_name = "gemini-1.5-flash"
embedding_model_name = "models/embedding-001"

set_verbose(False)


def reset_storage():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(UPLOAD_FOLDER)
    os.makedirs(CHROMA_DIR)


def ensure_directories_exist():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)


ensure_directories_exist()

# reset_storage()

# PDF size limit (e.g. 10MB)
MAX_PDF_SIZE = 10 * 1024 * 1024  # bytes


def process_pdf(pdf_path, lang="tr"):
    # Read PDF
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Request summary from Gemini
    model = genai.GenerativeModel(model_name)

    # Request summary and text analysis based on language selection
    if lang == "tr":
        summary_prompt = """
        Bu akademik makalenin kısa bir özetini çıkar. Önemli bulguları, araştırma yöntemlerini ve ana argümanları vurgula.
        Özet tek paragraf olmalı ve 3-4 cümleyi geçmemeli. Kullanılan modeller hakkında bilgi varsa bunu mutlaka içer.
        
        Makale:
        {}
        """.format(
            text[:7000]
        )  # Reading more content
    else:
        summary_prompt = """
        Create a concise summary of this academic paper. Highlight the important findings, research methods, and main arguments.
        The summary should be a single paragraph and not exceed 3-4 sentences. If there is information about models used, please include it.
        
        Paper:
        {}
        """.format(
            text[:7000]
        )  # Reading more content

    summary_response = model.generate_content(summary_prompt)
    summary = summary_response.text

    # Split text into more meaningful chunks
    # Adjust chunk_size based on PDF size
    file_size = os.path.getsize(pdf_path)
    if file_size > 5 * 1024 * 1024:  # Larger than 5MB
        chunk_size = 1000
    else:
        chunk_size = 1500

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=300,  # More overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Split at paragraph and line breaks
    )
    chunks = text_splitter.split_text(text)

    # Special analysis of PDF content
    models_used = extract_models_from_text(text, lang)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name, google_api_key=GOOGLE_API_KEY
    )

    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=CHROMA_DIR)

    return vectorstore, summary


def extract_models_from_text(text, lang):
    """Extracts models used from the text"""
    model = genai.GenerativeModel(model_name)

    if lang == "tr":
        prompt = """
        Aşağıdaki akademik makale metninden, kullanılan algoritma, model ve teknik isimleri tespit et.
        Sadece metinde geçen spesifik metotları listele. Tahmin yürütme.
        
        Metin:
        {}
        """.format(
            text[:10000]
        )
    else:
        prompt = """
        From the following academic paper text, identify algorithm, model, and technical method names used.
        Only list specific methods mentioned in the text. Don't speculate.
        
        Text:
        {}
        """.format(
            text[:10000]
        )

    response = model.generate_content(prompt)
    return response.text


@app.route("/", methods=["GET", "POST"])
def upload_file():
    # Get lang from form in POST, from query string in GET
    if request.method == "POST":
        lang = request.form.get("lang", "tr")
    else:
        lang = request.args.get("lang", "tr")

    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "Dosya seçilmedi" if lang == "tr" else "No file selected"

        file = request.files["pdf_file"]
        if file.filename == "":
            return "Dosya seçilmedi" if lang == "tr" else "No file selected"

        if file and file.filename.endswith(".pdf"):
            # Check file size
            if file.content_length > MAX_PDF_SIZE:
                return "Dosya boyutu çok büyük" if lang == "tr" else "File is too large"

            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process PDF and create vector database
            vectorstore, summary = process_pdf(filepath, lang)

            # Pass lang parameter to viewer.html
            return render_template(
                "viewer.html", filename=filename, summary=summary, lang=lang
            )

    return render_template("index.html", lang=lang)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    lang = data.get("lang", "tr")

    # Load Chroma database
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name, google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Answer the question
    if lang == "tr":
        system_prompt = """
        Sen bir akademik makale uzmanı AI asistansın. Makalenin içeriğine sadık kalarak soruları yanıtla.
        Eğer makale içinde bir bilgi yoksa, bunu açıkça belirt ve tahmin yürütme.
        Makalede kullanılan modeller, algoritmalar ve teknikler hakkında sorulursa, makaledeki bilgilere dayanarak detaylı cevap ver.
        """
    else:
        system_prompt = """
        You are an AI assistant specializing in academic papers. Answer questions strictly based on the paper's content.
        If information is not present in the paper, clearly state this and don't speculate.
        When asked about models, algorithms, or techniques used in the paper, provide detailed answers based on the information in the paper.
        """

    # Get more accurate answers by adding system prompt to the Gemini model
    llm = ChatGoogleGenerativeAI(
        model=model_name, google_api_key=GOOGLE_API_KEY, system_prompt=system_prompt
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True
    )

    # Answer the question
    result = qa_chain.invoke({"question": question, "chat_history": []})

    answer = result["answer"]
    return jsonify({"answer": answer, "lang": lang})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


@app.route("/change-language/<lang>")
def change_language(lang):
    session["language"] = lang
    return redirect(request.referrer or "/")


if __name__ == "__main__":
    app.run(debug=True)