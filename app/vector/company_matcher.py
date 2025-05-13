from typing import Optional, Tuple
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

# 임베딩 모델 초기화 (한글 지원 모델 선택)
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  
model = SentenceTransformer("jhgan/ko-sroberta-multitask")  

def normalize_name(name: str) -> str:
    """이름 문자열을 소문자로 변환하고, 영숫자 이외의 문자를 제거하여 정규화"""
    return ''.join(e for e in name.lower() if e.isalnum())

def is_fuzzy_match(name1: str, name2: str, threshold: int = 85) -> bool:
    """두 이름이 fuzzy string 유사도를 기준으로 일정 임계값 이상으로 유사한지 판단"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    return fuzz.ratio(norm1, norm2) >= threshold

def is_semantic_match(query_name: str, target_name: str, threshold: float = 0.90) -> bool:
    """두 이름의 의미적 유사도를 임베딩 후 cosine similarity로 계산하여 유사한지 판단"""
    emb_query = model.encode(query_name, convert_to_tensor=True)
    emb_target = model.encode(target_name, convert_to_tensor=True)
    score = util.cos_sim(emb_query, emb_target).item()
    return score >= threshold

def find_company_by_name(company_name: str, cur) -> Optional[Tuple[int, dict]]:
    """회사명을 기준으로 회사 정보를 조회하며, 없을 경우 제품명을 기반으로 fuzzy/semantic 검색 수행"""
    # 1차: 회사명 직접 매칭
    cur.execute("SELECT id, data FROM company WHERE name = %s", (company_name,))
    row = cur.fetchone()
    if row:
        return row

    # 2차: 제품명으로 fuzzy/semantic 유사도 검색
    logger.info(f"'{company_name}' 이름의 회사가 존재하지 않습니다. 제품명을 기준으로 fuzzy/semantic 검색 시도합니다.")
    cur.execute("SELECT id, data FROM company")
    candidates = cur.fetchall()

    for company_id, data in candidates:
        try:
            products = data.get("products", [])
            for product in products:
                product_name = product.get("name", "")
                if is_fuzzy_match(company_name, product_name):
                    logger.info(f"Fuzzy match 성공: '{company_name}' ≈ '{product_name}'")
                    return (company_id, data)
                if is_semantic_match(company_name, product_name):
                    logger.info(f"Semantic match 성공: '{company_name}' ≈ '{product_name}'")
                    return (company_id, data)
        except Exception as e:
            continue

    logger.warning(f"'{company_name}'에 해당하는 회사 또는 제품명을 찾지 못했습니다.")
    return None
