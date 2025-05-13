import os, faiss, logging
from sentence_transformers import SentenceTransformer
from app.db.db_config import connect_to_db
from app.vector.company_matcher import find_company_by_name
from app.vector.document_loader import load_company_documents, load_news_titles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 임베딩 모델 초기화 (한글 지원 모델 선택)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")  

def build_vector_index(documents: list):
    """문서 리스트를 임베딩하고 FAISS 인덱스를 생성"""
    vectors = model.encode(documents, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

def save_faiss_index(index, file_path: str):
    """FAISS 인덱스를 지정된 경로에 저장"""
    faiss.write_index(index, file_path)

def load_faiss_index(file_path: str):
    """저장된 FAISS 인덱스를 파일에서 불러오고 없으면 None을 반환"""
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    return None

def cache_index_for_company(company_id: int, cache_dir: str = "./faiss_cache"):
    """회사 뉴스에 대한 FAISS 인덱스를 캐싱하거나 새로 생성 후 저장. 인덱스와 뉴스 타이틀 리스트를 반환"""
    os.makedirs(cache_dir, exist_ok=True)
    index_file = os.path.join(cache_dir, f"{company_id}_index.faiss")

    # Load index if exists
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        news_titles = load_news_titles(company_id)
        logger.info(f"Loaded FAISS index for company {company_id}")
        return index, news_titles

    # Otherwise build
    news_titles = load_news_titles(company_id)
    if not news_titles:
        logger.warning(f"⚠️ 회사 {company_id}의 뉴스가 없습니다. 빈 벡터 인덱스를 반환합니다.")
        return None, []

    index, _ = build_vector_index(news_titles)
    faiss.write_index(index, index_file)
    logger.info(f"Saved FAISS index for company {company_id}")
    return index, news_titles

def search_similar_docs(query_text: str, index, documents: list, top_k: int = 5):
    """쿼리 문장을 임베딩하여 FAISS 인덱스에서 유사한 문서를 Top-K만큼 검색"""
    query_vector = model.encode(query_text, convert_to_numpy=True)
    query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]

def generate_vector_query(position):
    """주어진 이력 정보를 바탕으로 벡터 검색용 자연어 질의 문장을 생성"""
    company = position["company"]
    title = position["title"]
    start = position["start"]
    end = position.get("end") or "현재"

    return f"{company}에서 {start}부터 {end}까지 {title} 직무를 수행한 사람이 주도했을 만한 프로젝트나 경험에 대한 정보"

def vector_search_for_position(position: dict):
    """하나의 경력 포지션에 대해 회사 정보와 뉴스 기반의 벡터 검색을 수행하고 결과를 반환"""
    query = generate_vector_query(position)
    company = position["company"]

    conn = connect_to_db()
    cur = conn.cursor()
    result = find_company_by_name(company, cur)
    cur.close()
    conn.close()
    if result is None:
        return []

    company_id, company_data = result

    # ✅ 재직기간 내 회사 정보만 가져오기 (Always included)
    company_docs, _ = load_company_documents(company_id, company_data, position)

    # ✅ 벡터 인덱스: 뉴스만
    index, news_titles = cache_index_for_company(company_id)
    
    if index is None or not news_titles:
        news_docs = []
    else:
        news_docs = search_similar_docs(query, index, news_titles, top_k=5)
    
    return {
        "company_info": company_docs,
        "news_info": news_docs
    }