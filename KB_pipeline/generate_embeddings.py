from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_embeddings(text, model_name="NovaSearch/stella_en_1.5B_v5"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)

    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    vector_store = FAISS.from_texts(texts, embeddings_model)

    return vector_store, texts

def save_embeddings(vector_store, path):
    vector_store.save_local(path)

if __name__ == "__main__":
    pdf_path = "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/data/raw/KB-2.pdf"
    embeddings_path = "KB-2_vector_store.faiss"
    text_file_path = "KB-2.txt"

    text = extract_text_from_pdf(pdf_path)

    with open(text_file_path, "w") as text_file:
        text_file.write(text)

    vector_store, _ = generate_embeddings(text)

    save_embeddings(vector_store, embeddings_path)

    print(f"Text saved to {text_file_path}")
    print(f"Embeddings saved to {embeddings_path}")

