import json, os, logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.vector.vectorstore import vector_search_for_position
from app.preprocess.preprocess_talent import preprocess_talent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

def build_llm_input(talent_data: dict, vector_results_per_position: list[dict]) -> dict:
    preprocessed_positions = preprocess_talent(talent_data)

    positions = []
    for pos, vr in zip(preprocessed_positions, vector_results_per_position):
        if not isinstance(vr, dict):
            positions.append({
                "company": pos["company"],
                "title": pos["title"],
                "start": pos["start"],
                "end": pos["end"]
            })
            continue  # skip invalid result

        positions.append({
            "company": pos["company"],
            "title": pos["title"],
            "start": pos["start"],
            "end": pos["end"],
            "related_news": vr.get("news_info", []),
            "company_info": vr.get("company_info", [])
        })

    return {
        "summary": talent_data.get("summary", ""),
        "educations": talent_data.get("educations", []),
        "positions": positions
    }

def infer_experience_tags_from_talent(talent_data: dict) -> str:
    # 1. Vector search for each position
    vector_results = []
    positions = preprocess_talent(talent_data)

    for position in positions:
        result = vector_search_for_position(position)
        vector_results.append(result)
    # logger.info(f"🔍 벡터 검색 결과: {vector_results}")

    # 2. Build LLM input JSON
    llm_input = build_llm_input(talent_data, vector_results)

    # 3. Build prompt and call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_HOST")

    text_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    prompt = f"""You are an expert in analyzing a person's work experience based on the company, employment period, and job title, and inferring what kind of experience and competencies they are likely to have developed.

Below is the profile of a candidate, including their career history. For each role, additional context is provided through:

- Relevant **news headlines** (you may disregard any that are not related to the candidate's employment period or not related to the candidate's experience), and
- **Company data**, such as changes in employee count, capital changes, and investment rounds.

### Task:
Based on the candidate's work history and the context of each company during their employment period, infer what kinds of **experiences** and **competencies** this person is likely to have gained.

Return the result as **concise, simplified "experience tags" IN KOREAN**, such as:
- Use **1 to 5 words per tag**
- Avoid full sentences or explanatory phrases (short optional clarifications in parentheses are acceptable)
- Focus on essential and recognizable experience keywords
- Write each tag as a bullet point

Examples of well-formed tags:
- 성장기스타트업 경험 (토스 16년도 부터 19년까지 투자 규모 2배 이상 성장, 조직 규모 2배 이상 성장)  
- M&A 경험  
- 대규모 회사 경험 (삼성전자, SKT)
- 리더쉽 (타이틀, CPO 경험 다수, 창업)
- 상위권 대학교 (연세대학교)

You may infer from well-known company events only if they are **reasonably relevant to the candidate’s role** (e.g., if the person was CFO and the company went IPO — include it; if it was unrelated — ignore it).
If the candidate has a degree from a top-tier university (e.g., 서울대, 연세대), include a tag like “상위권 대학교 (연세대학교)”.
Return **at least 4 tags** and keep them concise and standardized.
        
        ### Output Examples
        #### Example 1
        ### output (Experience tags):
        - 초기 스타트업 경험
        - 빅테크 경험
        - 글로벌 런칭 경험
        - 기술 리더십 경험
        - B2C/B2B 도메인 경험
        - 물류 도메인 경험

        #### Example 2
        ### output:
        - 상위권대학교 (연세대학교)
        - 성장기스타트업 경험 (토스 16년도 부터 19년까지 투자 규모 2배 이상 성장, 조직 규모 2배 이상 성장)
        - 리더쉽 (타이틀, 챕터 리드, 테크 리드 기반으로 추론)
        - 대용량데이터처리경험 (네이버 AI가 21년 부터 하이퍼크로바까지 NLP 관련 처리 많이함)

        #### Example 3
        ### input:
        - 'educations': 'schoolName': '서울대학교', "degreeName": "Master of Business Administration (MBA)",
        - 'positions': 'company': 'company name2', 'title': 'Team Lead: Growth Strategy Team, Media Business Unit, 2025\n\nTeam Lead: Business Strategy Team, CSO Department, 2024\n- Analyze KT and its group companies' businesses, performing portfolio adjustment tasks, with a particular focus on enterprise and media/content sectors.', 'start': '2023-10', 'end': '2024-09',
                     'company': 'company name3', 'title': 'Chief Financial Officer', 'start': '2022-12', 'end': '2023-10', ...
        - 'related_news': [...]
        
        ### output:
        - 상위권대학교 (서울대학교)
        - 대규모 회사 경험 (KT 전략기획실, KT 미디어 사업부 성장전략 팀장 경험)
        - 리더쉽 (타이틀, CFO, 성장전략 팀장 경험 기반으로 추론)
        - IPO (밀리의 서재 CFO 재직 당시, 밀리의 서재가 IPO를 함)
        - M&A 경험 (지니뮤직에서 밀리의 서재 인수 리드)

        #### Example 4
        ### input:
        - 'educations': 'schoolName': '연세대학교',
        - 'positions': 'company': 'company name1', 'title': '공동대표/CPO', 'start': '2024-03', 'end': None,
                     'company': 'company name2', 'title': 'Entrepreneur', 'start': '2023-10', 'end': '2024-09',
                     'company': 'company name3', 'title': 'CPO', 'start': '2022-12', 'end': '2023-10', ...
        - 'related_news': [...]
        
        ### output:
        - 상위권대학교 (연세대학교)
        - 대규모 회사 경험 (삼성전자, SKT)
        - M&A 경험 (요기요 재직 중 사모펀드에 매각)
        - 리더쉽 (타이틀, CPO 경험 다수, 창업)
        - 신규 투자 유치 경험 (C level 재직기간 중 회사 신규 투자유치, Kasa Korea, LBox)

        ---
        ### Input (Candidate data and company information during employment):
        ```json
        {llm_input}
        ```
        
        ### output:
    """

    response = text_llm.invoke(prompt)
    return response.content

# 예시 테스트 코드
if __name__ == "__main__":
    with open("../../example_datas/talent_ex2.json", "r", encoding="utf-8") as f:
        talent_data = json.load(f)

    # LLM 추론
    result = infer_experience_tags_from_talent(talent_data)