# export_ax_oct.py
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

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

KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def ensure_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


KEYWORD = "AX"
ROWS_PER_PAGE = 50  # 최대 999까지 지원
DATE_FMT = "%Y%m%d%H%M"
CHUNK_DAYS = 3


def extract_bid_ordinal(value) -> tuple[str, int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "", 0
    if isinstance(value, str):
        cleaned = value.strip()
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        order_val = int(digits) if digits else 0
        return cleaned, order_val
    try:
        order_val = int(value)
    except (TypeError, ValueError):
        return "", 0
    return f"{order_val:03d}", order_val


def parse_doc_id(doc_id: str) -> tuple[str, str, int]:
    if "-" not in doc_id:
        return doc_id, "", 0
    base, suffix = doc_id.rsplit("-", 1)
    order_key, order_val = extract_bid_ordinal(suffix)
    return base, order_key, order_val

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
    secrets_available = False
    firebase_secret = None
    try:
        import streamlit as st

        firebase_secret = st.secrets.get("firebase")
        secrets_available = firebase_secret is not None
    except Exception:
        secrets_available = False

    if not FIREBASE_CRED_PATH.exists() and not secrets_available:
        raise FileNotFoundError("Firebase 자격 증명을 찾을 수 없습니다. secrets 또는 JSON 경로를 확인하세요.")
    if not firebase_admin._apps:
        if FIREBASE_CRED_PATH.exists():
            cred = credentials.Certificate(str(FIREBASE_CRED_PATH))
        else:
            cred = credentials.Certificate(dict(firebase_secret))
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
    end_dt = now_kst().replace(second=0, microsecond=0)
    if default_start is None:
        default_start = datetime(2025, 1, 1, tzinfo=KST)
    else:
        default_start = ensure_kst(default_start)

    start_dt = default_start
    try:
        client = db or init_firestore()
        latest_dt = get_latest_bid_datetime(client)
        if latest_dt:
            latest_dt = ensure_kst(latest_dt)
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
            print(f" Firestore 최신 데이터 조회 실패: {exc}")
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


def select_latest_variants(
    records: list[dict],
) -> tuple[list[dict], dict[str, set[str]], dict[str, int], dict[str, str]]:
    latest: dict[str, dict] = {}
    orders_to_remove: dict[str, set[str]] = defaultdict(set)
    extras: list[dict] = []

    for record in records:
        base_no = str(record.get("bidNtceNo") or "").strip()
        if not base_no:
            extras.append(record)
            continue

        order_raw = record.get("bidNtceOrd")
        order_key, order_value = extract_bid_ordinal(order_raw)

        entry = latest.get(base_no)
        if entry is None:
            latest[base_no] = {
                "record": record,
                "order_val": order_value,
                "order_key": order_key,
            }
        else:
            if order_value > entry["order_val"]:
                if entry["order_key"] != "":
                    orders_to_remove[base_no].add(entry["order_key"])
                entry["record"] = record
                entry["order_val"] = order_value
                entry["order_key"] = order_key
            else:
                if order_key != "":
                    orders_to_remove[base_no].add(order_key)

    keep_records = [info["record"] for info in latest.values()]
    keep_records.extend(extras)

    max_orders = {base: info["order_val"] for base, info in latest.items()}
    keep_order_keys = {base: info["order_key"] for base, info in latest.items()}

    return keep_records, orders_to_remove, max_orders, keep_order_keys


def cleanup_existing_variants(
    client: firestore.Client,
    *,
    verbose: bool = True,
) -> int:
    docs = client.collection(FIREBASE_COLLECTION).stream()
    grouped: dict[str, list[tuple[int, str, firestore.DocumentSnapshot]]] = defaultdict(list)
    for doc in docs:
        base_no, order_key, order_val = parse_doc_id(doc.id)
        grouped[base_no].append((order_val, order_key, doc))

    removed = 0
    for base_no, entries in grouped.items():
        if len(entries) <= 1:
            continue
        entries.sort(key=lambda item: (item[0], item[1]))
        keep = entries[-1][2].id
        for order_val, order_key, doc in entries[:-1]:
            try:
                doc.reference.delete()
                removed += 1
                if verbose:
                    print(f"  정리 삭제: {doc.id} (기준 {keep})")
            except Exception as exc:
                if verbose:
                    print(f"   정리 실패 {doc.id}: {exc}")
    return removed


def upsert_firestore(
    records: list[dict],
    db: firestore.Client | None = None,
    *,
    verbose: bool = True,
    collected_at: datetime | None = None,
    order_cleanup: dict[str, set[str]] | None = None,
    max_orders: dict[str, int] | None = None,
    keep_order_keys: dict[str, str] | None = None,
) -> int:
    if not records:
        if verbose:
            print("Firestore에 적재할 데이터가 없습니다.")
        return 0

    client = db or init_firestore()
    batch = client.batch()
    total = len(records)
    collected_at_dt = ensure_kst(collected_at) if collected_at else now_kst()
    collected_at_iso = collected_at_dt.isoformat()

    for idx, record in enumerate(records, start=1):
        normalized = normalize_record(record)
        normalized["collectedAt"] = collected_at_iso
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

    if order_cleanup:
        for base_no, orders in order_cleanup.items():
            for order_key in orders:
                doc_id = f"{base_no}-{order_key}".strip("-")
                if not doc_id:
                    continue
                try:
                    client.collection(FIREBASE_COLLECTION).document(doc_id).delete()
                    if verbose:
                        print(f"  삭제: {doc_id}")
                except Exception as exc:
                    if verbose:
                        print(f"  삭제 실패 {doc_id}: {exc}")

    if max_orders and keep_order_keys:
        for base_no, max_order in max_orders.items():
            if not base_no:
                continue
            try:
                docs = (
                    client.collection(FIREBASE_COLLECTION)
                    .where("bidNtceNo", "==", base_no)
                    .stream()
                )
                for doc in docs:
                    data = doc.to_dict()
                    order_key, order_val = extract_bid_ordinal(data.get("bidNtceOrd"))
                    if order_val < max_order or order_key != keep_order_keys.get(base_no):
                        if order_key == keep_order_keys.get(base_no):
                            continue
                        doc.reference.delete()
                        if verbose:
                            print(f"  정리 삭제: {doc.id}")
            except Exception as exc:
                if verbose:
                    print(f"  ⚠️ 정리 실패 ({base_no}): {exc}")

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
        "collected_at": None,
        "cleanup_removed": 0,
        "meta_updated": False,
    }

    if start_dt >= end_dt:
        if verbose:
            print("새로 수집할 데이터가 없습니다.")
        collected_at = now_kst()
        summary["collected_at"] = collected_at
        cleanup_removed = cleanup_existing_variants(db, verbose=verbose)
        summary["cleanup_removed"] = cleanup_removed
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
                        print(f" {msg}")
                    raise RuntimeError(msg)
                elif verbose:
                    print(" 공고명 파라미터 서버 필터 확인 완료.")

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

    deduped_records, orders_to_remove, max_orders, keep_order_keys = select_latest_variants(filtered_records)

    collected_at = now_kst()
    summary["collected_at"] = collected_at

    if deduped_records:
        upserted = upsert_firestore(
            deduped_records,
            db=db,
            verbose=verbose,
            collected_at=collected_at,
            order_cleanup=orders_to_remove,
            max_orders=max_orders,
            keep_order_keys=keep_order_keys,
        )
    else:
        upserted = 0

    summary["upserted_records"] = upserted

    cleanup_removed = cleanup_existing_variants(db, verbose=verbose)
    summary["cleanup_removed"] = cleanup_removed

    if upserted > 0:
        meta_ref = db.collection(FIREBASE_META_COLLECTION).document(FIREBASE_META_DOC)
        meta_ref.set(
            {
                "collectedDate": collected_at.date().isoformat(),
                "collectedAt": collected_at.isoformat(),
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
        print(f"수집 실패: {exc}")
        sys.exit(1)

    print(
        "수집 완료: "
        f"{result['start_dt'].strftime(DATE_FMT)} ~ {result['end_dt'].strftime(DATE_FMT)}, "
        f"총 {result['total_collected']}건 수신, "
        f"{result['filtered_records']}건 필터링, "
        f"{result['upserted_records']}건 업서트"
    )

if __name__ == "__main__":
    main()