# export_ax_oct.py
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

BASE_URL = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoServcPPSSrch"
load_dotenv()


def _decode_service_key(key: str) -> str:
    return unquote(key) if "%" in key else key


def resolve_service_key() -> str | None:
    env_key = os.getenv("G2B_SERVICE_KEY")
    if env_key:
        return _decode_service_key(env_key)

    try:
        import streamlit as st

        secret_key = st.secrets.get("G2B_SERVICE_KEY")
        if secret_key:
            return _decode_service_key(secret_key)
    except Exception:
        pass

    return None


SERVICE_KEY = resolve_service_key()
FIREBASE_CRED_PATH = Path(
    os.getenv(
        "G2B_FIREBASE_CRED_PATH",
        Path(__file__).resolve().parent / "g2b-bid-finder-firebase-adminsdk-fbsvc-aae6f1c96d.json",
    )
)
FIREBASE_COLLECTION = "bid_pblanc_list"
FIREBASE_META_COLLECTION = "meta"
FIREBASE_META_DOC = "collection_state"

KEYWORD = "AX"
ROWS_PER_PAGE = 50  # 최대 999까지 지원
DATE_FMT = "%Y%m%d%H%M"
CHUNK_DAYS = 3

def fetch_page(page: int, begin: str, end: str, keyword: str | None = None) -> list[dict]:
    params = {
        "serviceKey": SERVICE_KEY,
        "ServiceKey": SERVICE_KEY,  # 일부 엔드포인트는 대소문자를 구분하지 않으나 대비
        "pageNo": page,
        "numOfRows": ROWS_PER_PAGE,
        "type": "json",
        "inqryDiv": "1",            # 1: 등록일시
        "inqryBgnDt": begin,
        "inqryEndDt": end,
    }
    if keyword:
        params["bidNtceNm"] = keyword
    r = requests.get(BASE_URL, params=params, timeout=20)
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else ""
        raise requests.HTTPError(f"{exc} | response: {detail}") from exc
    payload = r.json()

    if payload["response"]["header"]["resultCode"] != "00":
        raise RuntimeError(payload["response"]["header"]["resultMsg"])

    items = payload["response"]["body"].get("items")
    if not items:
        return []
    # items는 dict 또는 list(dict)로 내려옵니다.
    if isinstance(items, dict):
        return [items]
    return items

def test_keyword_filter(sample_items: list[dict], keyword: str) -> bool:
    if not sample_items:
        return False
    # 서버 필터가 적용됐다면 모든 항목이 키워드를 포함해야 함
    return all(keyword.lower() in (item.get("bidNtceNm") or "").lower() for item in sample_items)


def init_firestore():
    if not FIREBASE_CRED_PATH.exists() and "firebase" not in getattr(__import__("streamlit"), "secrets", {}):
        raise FileNotFoundError("Firebase 자격 증명을 찾을 수 없습니다. secrets 또는 JSON 경로를 확인하세요.")
    if not firebase_admin._apps:
        if FIREBASE_CRED_PATH.exists():
            cred = credentials.Certificate(str(FIREBASE_CRED_PATH))
        else:
            import streamlit as st

            cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def get_latest_bid_datetime(db) -> datetime | None:
    docs = (
        db.collection(FIREBASE_COLLECTION)
        .order_by("bidNtceDt", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        value = doc.to_dict().get("bidNtceDt")
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


def calc_period(
    default_start: datetime | None = None,
    db: firestore.Client | None = None,
    verbose: bool = True,
) -> tuple[datetime, datetime]:
    end_dt = datetime.now().replace(second=0, microsecond=0)
    if default_start is None:
        default_start = datetime(2025, 1, 1)

    start_dt = default_start
    try:
        client = db or init_firestore()
        latest_dt = get_latest_bid_datetime(client)
        if latest_dt:
            candidate = latest_dt + timedelta(seconds=1)
            if candidate <= end_dt:
                start_dt = candidate
                if verbose:
                    print(f"Firestore 최신 공고일시: {latest_dt.isoformat()} → {start_dt.isoformat()}부터 수집합니다.")
            else:
                start_dt = end_dt
                if verbose:
                    print("이미 최신 데이터가 수집되어 있습니다.")
        else:
            if verbose:
                print("Firestore에 기존 데이터가 없거나 최신 공고일시를 찾지 못했습니다. 기본 시작일을 사용합니다.")
    except Exception as exc:
        if verbose:
            print(f"⚠️ Firestore 최신 데이터 조회 실패: {exc}")
            print("기본 시작일을 사용합니다.")

    return start_dt, end_dt


def normalize_record(record: dict) -> dict:
    normalized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            normalized[key] = value.isoformat()
        elif isinstance(value, datetime):
            normalized[key] = value.isoformat()
        elif pd.isna(value):
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized


def upsert_firestore(
    records: list[dict],
    db: firestore.Client | None = None,
    *,
    verbose: bool = True,
) -> int:
    if not records:
        if verbose:
            print("Firestore에 적재할 데이터가 없습니다.")
        return 0

    client = db or init_firestore()
    batch = client.batch()
    total = len(records)
    collected_at = datetime.now().isoformat()

    for idx, record in enumerate(records, start=1):
        normalized = normalize_record(record)
        normalized["collectedAt"] = collected_at
        doc_id = f"{normalized.get('bidNtceNo', '')}-{normalized.get('bidNtceOrd', '')}".strip("-")
        if not doc_id:
            doc_id = normalized.get("untyNtceNo") or f"auto-{idx}"
        doc_ref = client.collection(FIREBASE_COLLECTION).document(doc_id)
        batch.set(doc_ref, normalized, merge=True)
        if verbose:
            print(f"  [{idx}/{total}] {normalized.get('bidNtceDt')} | {normalized.get('bidNtceNm')}")

        if idx % 400 == 0:
            batch.commit()
            if verbose:
                print(f"  Firestore 배치 커밋 완료 ({idx}건)")
            batch = client.batch()

    batch.commit()
    if verbose:
        print(f"Firestore 적재 완료: 총 {total}건")
    return total


def collect_and_upsert(
    keyword: str = KEYWORD,
    *,
    verbose: bool = True,
) -> dict:
    if not SERVICE_KEY:
        raise RuntimeError("환경변수 G2B_SERVICE_KEY를 설정해주세요.")

    db = init_firestore()
    start_dt, end_dt = calc_period(db=db, verbose=verbose)

    summary = {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "total_collected": 0,
        "filtered_records": 0,
        "upserted_records": 0,
        "meta_updated": False,
    }

    if start_dt >= end_dt:
        if verbose:
            print("새로 수집할 데이터가 없습니다.")
        return summary

    if verbose:
        print(f"데이터 수집 기간: {start_dt.strftime(DATE_FMT)} ~ {end_dt.strftime(DATE_FMT)}")
        print("데이터 수집 시작...")

    collected: list[dict] = []
    chunk_start = start_dt
    keyword_tested = False

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS) - timedelta(minutes=1), end_dt)
        begin = chunk_start.strftime(DATE_FMT)
        end = chunk_end.strftime(DATE_FMT)

        if verbose:
            print(f"[{begin} ~ {end}] 구간 요청 시작")

        page = 1
        while True:
            rows = fetch_page(page, begin, end, keyword)
            if not rows:
                break
            if page == 1 and not keyword_tested:
                keyword_tested = True
                if not test_keyword_filter(rows, keyword):
                    msg = "공고명 파라미터로 서버 필터링이 되지 않습니다. 수집을 중단합니다."
                    if verbose:
                        print(f"⚠️ {msg}")
                    raise RuntimeError(msg)
                elif verbose:
                    print("✅ 공고명 파라미터 서버 필터 확인 완료.")

            collected.extend(rows)
            if verbose:
                print(f"  {page} 페이지 수신 (누적 {len(collected)}건)")
            page += 1
            time.sleep(0.15)  # API 요청 제한 완화용 딜레이

        chunk_start = chunk_end + timedelta(minutes=1)

    summary["total_collected"] = len(collected)
    if verbose:
        print(f"총 {len(collected)}건 수신")

    if not collected:
        return summary

    filtered_records = [
        row for row in collected
        if keyword.lower() in (row.get("bidNtceNm") or "").lower()
    ]
    summary["filtered_records"] = len(filtered_records)

    if verbose:
        print(f"필터링 후 {len(filtered_records)}건 남음")

    if not filtered_records:
        return summary

    upserted = upsert_firestore(filtered_records, db=db, verbose=verbose)
    summary["upserted_records"] = upserted

    if upserted > 0:
        meta_ref = db.collection(FIREBASE_META_COLLECTION).document(FIREBASE_META_DOC)
        meta_ref.set(
            {
                "collectedDate": summary["end_dt"].date().isoformat(),
                "collectedAt": datetime.now().isoformat(),
                "upsertedRecords": upserted,
            },
            merge=True,
        )
        summary["meta_updated"] = True

    return summary


def main():
    try:
        result = collect_and_upsert()
    except Exception as exc:
        print(f"❌ 수집 실패: {exc}")
        sys.exit(1)

    print(
        "✅ 수집 완료: "
        f"{result['start_dt'].strftime(DATE_FMT)} ~ {result['end_dt'].strftime(DATE_FMT)}, "
        f"총 {result['total_collected']}건 수신, "
        f"{result['filtered_records']}건 필터링, "
        f"{result['upserted_records']}건 업서트"
    )

if __name__ == "__main__":
    main()