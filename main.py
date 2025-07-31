from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import shutil

os.environ["GOOGLE_API_KEY"] = "AIzaSyCQtR3ZmOigAyJ4L-B8veRPekfA0kBcm44"

# LangChain + Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Rate Limiting + Backoff
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global Vectorstore for reuse
VECTORSTORE_PATH = "vectorstore_index"
vectorstore = None

# Retry handler
def backoff_handler(details):
    print(f"🔁 Retrying... attempt {details['tries']} after {details['wait']}s")

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
def generate_format(request: Request, raw_data: str = Form(...)):
    global vectorstore

    print("📥 Raw data received:", raw_data[:100] + "..." if len(raw_data) > 100 else raw_data)

    try:
        # Reuse vectorstore if available
        if vectorstore is None and os.path.exists(VECTORSTORE_PATH):
            try:
                embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedding, allow_dangerous_deserialization=True)
                print("✅ Vectorstore loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load vectorstore: {e}")

        if vectorstore:
            docs = vectorstore.similarity_search(raw_data, k=4)
            combined = "\n\n".join(doc.page_content for doc in docs)
            print(f"📄 Found {len(docs)} relevant documents")
        else:
            combined = raw_data
            print("📝 Using raw data (no vectorstore)")

        @on_exception(expo, RateLimitException, on_backoff=backoff_handler, max_tries=3)
        @limits(calls=1, period=15)
        def run_model_chain(raw_input):
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. Format the given raw college information under the following headings:

            - Executive Summary
            - Profile of the Institution
            - Major Achievements
            - Key Events
            - Outcomes

            Raw Input:
            {raw_input}
            """)
            llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)
            chain = RunnableSequence(prompt | llm | StrOutputParser())
            return chain.invoke({"raw_input": raw_input})

        formatted_output = run_model_chain(combined)
        print("✅ Text formatting completed successfully")

        return templates.TemplateResponse("form.html", {
            "request": request,
            "formatted_output": formatted_output,
            "raw_text": raw_data
        })

    except Exception as e:
        print("❌ Error occurred:", str(e))
        return templates.TemplateResponse("form.html", {
            "request": request,
            "formatted_output": f"Error occurred: {str(e)}",
            "raw_text": raw_data
        })

@app.post("/upload")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    global vectorstore
    
    try:
        print(f"📤 Uploading file: {pdf_file.filename}")
        print(f"📋 File type: {pdf_file.content_type}")
        print(f"📏 File size: {pdf_file.size if hasattr(pdf_file, 'size') else 'Unknown'}")
        
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are allowed"}
            )
        
        # Create upload directory
        upload_dir = Path("uploaded_files")
        upload_dir.mkdir(exist_ok=True)
        
        # Save the uploaded PDF
        file_location = upload_dir / pdf_file.filename
        contents = await pdf_file.read()
        
        with open(file_location, "wb") as f:
            f.write(contents)
        
        print(f"💾 File saved to: {file_location}")
        
        # Load and process PDF
        print("📖 Loading PDF...")
        loader = PyPDFLoader(str(file_location))
        pages = loader.load()
        print(f"📄 Loaded {len(pages)} pages from PDF")
        
        if not pages:
            return JSONResponse(
                status_code=400,
                content={"error": "PDF appears to be empty or corrupted"}
            )
        
        # Split documents
        print("✂️ Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)
        print(f"📝 Created {len(documents)} document chunks")
        
        # Create embeddings and vectorstore
        print("🔗 Creating embeddings...")
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents, embedding)
        
        # Save vectorstore
        print("💾 Saving vectorstore...")
        vectorstore.save_local(VECTORSTORE_PATH)
        
        print("✅ PDF processing completed successfully!")
        
        return JSONResponse(content={
            "message": f"✅ PDF '{pdf_file.filename}' uploaded and processed successfully!",
            "filename": pdf_file.filename,
            "pages": len(pages),
            "chunks": len(documents),
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to process PDF: {str(e)}",
                "status": "error"
            }
        )

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "AI Generator Agent is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)