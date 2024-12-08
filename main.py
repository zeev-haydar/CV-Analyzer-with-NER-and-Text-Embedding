from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
import logging
from fastapi.responses import JSONResponse
import pydantic
from fastapi.middleware.cors import CORSMiddleware
from models.IndonesianCVMatcher import IndonesianCVMatcher
import http
import PyPDF2
import mimetypes
from typing import Optional, Dict, Any
import traceback


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# initialize this class when the app startup

cv_matcher = IndonesianCVMatcher()

class BaseModel(pydantic.BaseModel):
    pass

class Message(BaseModel):
    status: int
    message: str
    payload: Optional[Dict[str, Any]] = None
    


def is_pdf(file):
    file_type, _ = mimetypes.guess_type(file.filename)
    return file_type == "application/pdf"

def serialize_result(result):
    if isinstance(result, dict):
        return {key: serialize_result(value) for key, value in result.items()}
    elif isinstance(result, set):
        return list(result)
    else:
        return result

@app.post(
    "/analyze_cv",
    responses={
        http.HTTPStatus.BAD_REQUEST: {"model": Message},
        http.HTTPStatus.INTERNAL_SERVER_ERROR: {"model": Message},
        http.HTTPStatus.OK: {"model": Message},
    },
)
async def analyze_cv(
    cv_file: UploadFile = File(...),
    job_requirements_text: str = Form(...),
):

    if cv_file is None:
        return JSONResponse(
            status_code=http.HTTPStatus.BAD_REQUEST,
            content=Message(
                status=http.HTTPStatus.BAD_REQUEST,
                message="No file uploaded",
                payload=None,
            ).model_dump(),
        )

    if not is_pdf(cv_file):
        return JSONResponse(
            status_code=http.HTTPStatus.BAD_REQUEST,
            content=Message(
                status=http.HTTPStatus.BAD_REQUEST,
                message="The file must be in PDF format",
                payload=None,
            ).model_dump(),
        )

    # Extract text from PDF
    try:
        pdf_reader = PyPDF2.PdfReader(cv_file.file)
        cv_text = " ".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return JSONResponse(
            status_code=http.HTTPStatus.BAD_REQUEST,
            content=Message(
                status=http.HTTPStatus.BAD_REQUEST,
                message=f"Error reading PDF: {str(e)}",
                payload=None,
            ).model_dump(),
        )

    # Get results
    try:
        key_result, result_text = cv_matcher.print_match_requirements(
            cv_text, job_requirements_text
        )
        key_result_serialized = serialize_result(key_result)
        logger.info(f"Key Results: {key_result}")
        logger.info(f"Text version: {result_text}")
        return JSONResponse(
            status_code=http.HTTPStatus.OK,
            content=Message(
                status=http.HTTPStatus.OK,
                message="CV analysis successful",
                payload={"key_result": key_result_serialized, "result_text": result_text},
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Error analyzing CV: {traceback.format_exc()}")
        return JSONResponse(
            status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR,
            content=Message(
                status=http.HTTPStatus.INTERNAL_SERVER_ERROR,
                message=f"Error analyzing CV: {str(e)}",
                payload=None,
            ).model_dump(), 
        )
