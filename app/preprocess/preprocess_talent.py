from datetime import datetime
from typing import List, Dict, Optional
import json

def preprocess_talent(talent_json: dict) -> List[Dict]:
    """
    인재 정보에서 회사, 재직기간, 직무(타이틀)을 추출하여 반환
    """
    processed = []
    positions = talent_json.get("positions", [])

    for pos in positions:
        company = pos.get("companyName")
        title = pos.get("title")
        start = pos["startEndDate"].get("start")
        end = pos["startEndDate"].get("end")  # may be None

        processed.append({
            "company": company,
            "title": title,
            "start": f'{start["year"]}-{start["month"]:02}',
            "end": f'{end["year"]}-{end["month"]:02}' if end else None
        })
    return processed