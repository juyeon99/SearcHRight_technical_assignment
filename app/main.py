from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import logging
from typing import List

from app.llm.llm_inference import infer_experience_tags_from_talent

app = FastAPI()
logger = logging.getLogger(__name__)

class TalentRequest(BaseModel):
    """지원자의 이력 데이터를 나타내는 모델 (JSON 바디 방식 입력 시 사용)"""
    talent: dict = Field(..., description="지원자의 이력 정보를 담은 딕셔너리")

class ExperienceTagsResponse(BaseModel):
    """경험 태그 추론 결과를 나타내는 응답 모델"""
    experience_tags: List[str] = Field(..., description="LLM이 추론한 경험 태그 리스트")

def parse_experience_tags(raw_output: str) -> list[str]:
    """
    LLM 출력에서 경험 태그 항목들을 리스트로 추출합니다.
    - "-", "•", 숫자형 등으로 시작하는 항목도 처리합니다.
    - 불필요한 접두 문구 제거 시도도 포함합니다.
    """
    lines = raw_output.splitlines()

    parsed = []
    for line in lines:
        line = line.strip()

        # 리스트 형태 문장으로 보이는 항목 필터링
        if line.startswith(("-", "•", "*")) or (line[:2].isdigit() and line[2:3] in [".", ")"]):
            parsed.append(line.lstrip("-•*0123456789. )").strip())

    # 혹시라도 아무 리스트 항목도 없으면 줄 단위로라도 리턴
    if not parsed:
        parsed = [line.strip() for line in lines if line.strip()]

    return parsed

@app.post(
    "/infer-tags/upload/",
    summary="JSON 파일을 업로드하여 경험 태그 추론",
    description="""
    인재 데이터를 담은 JSON 파일을 업로드받아, 
    LLM을 통해 해당 인재의 경력 기반 경험 태그를 추론합니다.
    
    input 파일은 JSON 형식이어야 합니다.
    """,
    response_model=ExperienceTagsResponse,
    responses={
        200: {"description": "성공적으로 경험 태그를 추론함"},
        400: {"description": "입력 파일 형식이 잘못됨"},
        422: {"description": "요청 형식이 잘못되었거나 파일이 누락됨"},
        500: {"description": "태그 추론 중 내부 오류 발생"},
    },
)
async def infer_experience_tags_from_file(file: UploadFile = File(...)):
    """
    JSON 파일에서 경험 태그를 추론합니다.

    - **file**: 인재 정보 JSON 파일
    - **반환값**: LLM이 추론한 경험 태그 리스트
    """
    try:
        content = await file.read()
        talent_data = json.loads(content)
        raw_result = infer_experience_tags_from_talent(talent_data)

        # 문자열일 경우 파싱
        if isinstance(raw_result, str):
            experience_tags = parse_experience_tags(raw_result)
        elif isinstance(raw_result, list):
            experience_tags = raw_result
        else:
            experience_tags = [str(raw_result)]

        return JSONResponse(content={"experience_tags": experience_tags}, status_code=200)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효하지 않은 JSON 형식입니다.")
    except Exception as e:
        logger.exception("파일 기반 LLM 추론 실패")
        raise HTTPException(status_code=500, detail=str(e))