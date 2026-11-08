
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mat_ai_digit_rubber_integration import process_question as run_pipeline


app = FastAPI(
    title="DigiRubber Entity and Relationship Extraction API"
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/process-question")
def process_question_endpoint(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty."
        )

    try:
        return run_pipeline(request.question)

    except Exception as exc:
        logging.exception("Question processing failed.")
        raise HTTPException(
            status_code=500,
            detail="Question processing failed."
        ) from exc
