from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reasoning_agent import ReasoningAgent
from file_extraction import extract_text

app = FastAPI(title="AI vs Human Text Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ReasoningAgent("classifier.joblib", "vectorizer.joblib")


@app.get("/")
def serve_frontend():
    return FileResponse("wordprint.html")


class TextInput(BaseModel):
    text: str


@app.post("/analyze")
def analyze(payload: TextInput):
    if len(payload.text.split()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Dumb!! Text is too short for reliable analysis. Please enter at least 50 words.",
        )
    return agent.predict(payload.text)


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        text = extract_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    text = text.strip()
    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough readable text from this file.",
        )

    result = agent.predict_document(text)
    result["extracted_text_preview"] = text[:500]
    return result


@app.get("/health")
def health():
    return {"status": "ok cutie pie"}