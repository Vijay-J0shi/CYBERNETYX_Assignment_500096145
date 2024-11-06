import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil
from chromadb import Client
from sentence_transformers import SentenceTransformer


nest_asyncio.apply()

app = FastAPI()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
client = Client() 
collection = client.get_or_create_collection("text_embeddings")  


@app.get("/")
def main():
    content = """
    <html>
        <body>
            <form action="/uploadfile/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".txt"/>
                <input type="submit"/>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
  
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    
    embedding = embedding_model.encode(text).tolist()

  
    doc_id = f"doc_{file.filename}"
    collection.add(documents=[text], metadatas=[{"filename": file.filename}], ids=[doc_id], embeddings=[embedding])

    return {"filename": file.filename, "message": "File uploaded and embedding stored successfully!"}


