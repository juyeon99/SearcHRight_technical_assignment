import logging
from datetime import datetime, date
from typing import Optional
from app.db.db_config import connect_to_db

logger = logging.getLogger(__name__)

def parse_date_or_none(date_str: str, fmt: str) -> Optional[date]:
    """문자열을 날짜 객체로 파싱, 실패할 경우 None을 반환"""
    try:
        return datetime.strptime(date_str, fmt).date()
    except:
        return None

def is_within_period(target: date, start: date, end: date) -> bool:
    """특정 날짜가 주어진 시작일과 종료일 사이에 포함되는지 여부를 반환"""
    return start <= target <= end

def load_company_documents(company_id: int, company_data: dict, position: Optional[dict]):
    """주어진 재직 기간 내의 회사 정보를 로딩하고, 관련 뉴스 타이틀을 함께 반환"""
    docs = []
    docs.append(f"회사명: {position['company']}")
    
    # 날짜 파싱
    if position:
        emp_start_date = parse_date_or_none(position["start"], "%Y-%m")
        emp_end_date = parse_date_or_none(position["end"], "%Y-%m") if position.get("end") else date.today()
        docs.append(f"재직 기간: {position['start']} ~ {position['end'] if position.get('end') else '현재'}")
    else:
        emp_start_date = date.min
        emp_end_date = date.max
    
    conn = connect_to_db()
    cur = conn.cursor()

    # 회사 뉴스 조회
    cur.execute("SELECT title, news_date FROM company_news WHERE company_id = %s", (company_id,))
    news_docs = [f"{title} ({news_date.strftime('%Y-%m-%d')})" for title, news_date in cur.fetchall()]

    cur.close()
    conn.close()

    # 🎯 투자 정보
    if "investment" in company_data:
        for inv in company_data.get("investment", {}).get("data", []):
            invest_at = parse_date_or_none(inv.get("investAt", ""), "%Y-%m-%d")
            if invest_at and is_within_period(invest_at, emp_start_date, emp_end_date):
                investmentAmount = inv.get("investmentAmount", "")
                investor_names = ", ".join(i.get("name", "") for i in inv.get("investor", []))
                docs.append(f"{inv['investAt']}; {inv['level']}; 총 투자 금액: {investmentAmount}; 투자 유치: {investor_names}")

    # 🎯 자본금 변동
    if "finance" in company_data:
        for item in company_data.get("finance", {}).get("data", []):
            year = item.get("year")
            if year:
                finance_date = date(year, 1, 1)
                if is_within_period(finance_date, emp_start_date, emp_end_date):
                    docs.append(f"{year}년 자본금: {item.get('capital', '')}, 순이익: {item.get('netProfit', '')}")

    # 🎯 재직자 수 변동
    if "organization" in company_data:
        for item in company_data.get("organization", {}).get("data", []):
            ref_month = parse_date_or_none(item.get("referenceMonth", ""), "%Y-%m")
            if ref_month and is_within_period(ref_month, emp_start_date, emp_end_date):
                docs.append(f"{item.get('referenceMonth')} 기준 인원수: {item.get('value')}명")

    # 회사 소개
    if "base_company_info" in company_data:
        try:
            intro = company_data["base_company_info"]["data"]["seedCorp"]["corpIntroKr"]
            docs.append(f"회사 소개: {intro}")
        except Exception:
            pass
    
    return docs, news_docs

def load_news_titles(company_id: int) -> list[str]:
    """회사의 뉴스 제목과 날짜를 불러와 정렬된 리스트로 반환"""
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT title, news_date
        FROM company_news
        WHERE company_id = %s
        ORDER BY news_date ASC
    """, (company_id,))
    news_titles = [f"{title} ({news_date.strftime('%Y-%m-%d')})" for title, news_date in cur.fetchall()]
    cur.close()
    conn.close()
    return news_titles
