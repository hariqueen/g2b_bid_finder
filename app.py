
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import firebase_admin
from firebase_admin import credentials, firestore

from export_ax_oct import collect_and_upsert

st.set_page_config(
    page_title="나라장터 용역 입찰공고 모니터링",
    layout="wide"
)


FIREBASE_CRED_PATH = Path(
    os.getenv(
        "G2B_FIREBASE_CRED_PATH",
        Path(__file__).resolve().parent / "g2b-bid-finder-firebase-adminsdk-fbsvc-aae6f1c96d.json",
    )
)
FIREBASE_COLLECTION = os.getenv("G2B_FIREBASE_COLLECTION", "bid_pblanc_list")

def _has_secrets() -> bool:
    return FIREBASE_CRED_PATH.exists() or ("firebase" in st.secrets)

FIREBASE_ENABLED = _has_secrets()

KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def now_kst_naive() -> datetime:
    return now_kst().replace(tzinfo=None)


def get_firestore_client():
    if not FIREBASE_ENABLED:
        raise FileNotFoundError("Firebase credential file not found.")
    if not firebase_admin._apps:
        if FIREBASE_CRED_PATH.exists():
            cred = credentials.Certificate(str(FIREBASE_CRED_PATH))
        else:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def get_latest_bid_datetime() -> datetime | None:
    if not FIREBASE_ENABLED:
        return None
    try:
        client = get_firestore_client()
        docs = (
            client.collection(FIREBASE_COLLECTION)
            .order_by("bidNtceDt", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            data = doc.to_dict()
            bid_value = data.get("bidNtceDt")
            if isinstance(bid_value, datetime):
                return bid_value
            if isinstance(bid_value, str):
                try:
                    return datetime.fromisoformat(bid_value)
                except ValueError:
                    pass

            collected = data.get("collectedAt")
            if isinstance(collected, datetime):
                return collected
            if isinstance(collected, str):
                try:
                    return datetime.fromisoformat(collected)
                except ValueError:
                    pass
    except Exception:
        return None
    return None


def get_last_collection_date() -> date | None:
    if not FIREBASE_ENABLED:
        return None
    try:
        client = get_firestore_client()
        doc = (
            client.collection("meta")
            .document("collection_state")
            .get()
        )
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        collected_date = data.get("collectedDate")
        if isinstance(collected_date, str):
            try:
                return date.fromisoformat(collected_date)
            except ValueError:
                pass
        collected_at = data.get("collectedAt")
        if isinstance(collected_at, str):
            try:
                return datetime.fromisoformat(collected_at).date()
            except ValueError:
                pass
    except Exception:
        return None
    return None


@st.cache_data
def load_data() -> tuple[pd.DataFrame, str]:
    """
    Firestore에서 데이터를 조회하고, 실패 시 빈 DataFrame을 반환합니다.
    """
    records: list[dict] = []
    source = "firestore"

    if not FIREBASE_ENABLED:
        return pd.DataFrame(), "disabled"

    try:
        client = get_firestore_client()
        docs = client.collection(FIREBASE_COLLECTION).stream()
        records = [doc.to_dict() for doc in docs]
    except Exception as exc:
        return pd.DataFrame(), f"error: {exc}"

    if not records:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(records)

    date_cols = ["bidNtceDt", "bidBeginDt", "bidClseDt", "bidQlfctRgstDt"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["asignBdgtAmt", "presmptPrce"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    now = now_kst_naive()

    def compute_status(row):
        begin = row.get("bidBeginDt")
        close = row.get("bidClseDt")
        if pd.isna(begin) or pd.isna(close):
            return "정보 부족"
        if now < begin:
            return "입찰 전"
        if begin <= now <= close:
            return "진행중"
        if now > close:
            return "마감"
        return "정보 부족"

    if len(df) > 0:
        df["status"] = df.apply(compute_status, axis=1)
    else:
        df["status"] = []

    if "reNtceYn" in df.columns:
        df["reNtceYn"] = (
            df["reNtceYn"].fillna("").replace({"재공고": "Y", "최초공고": "N"})
        )
    else:
        df["reNtceYn"] = ""

    return df, source


def format_money(v):
    if pd.isna(v):
        return "-"
    try:
        v = float(v)
    except Exception:
        return str(v)

    def fmt_unit(amount: float) -> str:
        rounded = round(amount)
        if abs(amount - rounded) < 1e-6:
            return f"{int(rounded)}"
        return f"{amount:.1f}".rstrip("0").rstrip(".")

    if v >= 1_0000_0000:  # 억 단위 이상
        return f"{fmt_unit(v / 1_0000_0000)}억원"
    if v >= 1_0000:  # 만원 단위 이상
        return f"{fmt_unit(v / 1_0000)}만원"
    return f"{int(v):,}원"


@st.dialog("입찰공고 상세", width="large")
def show_detail_dialog(row: pd.Series):
    now = now_kst_naive()
    bid_begin = row.get("bidBeginDt")
    bid_close = row.get("bidClseDt")

    st.write(f"입찰공고일시: {row.get('bidNtceDt', '-')}")
    if pd.notna(bid_begin):
        st.write(f"입찰개시일시: {bid_begin}")
    else:
        st.write("입찰개시일시: -")

    if pd.notna(bid_close):
        st.write(f"입찰마감일시: {bid_close}")
        if bid_close > now and pd.notna(bid_begin) and bid_begin <= now:
            remaining = bid_close - now
            days = remaining.days
            hours = remaining.seconds // 3600
            minutes = (remaining.seconds % 3600) // 60
            st.write(f"입찰 마감까지 남은시간: {days}일 {hours}시간 {minutes}분")
    else:
        st.write("입찰마감일시: -")

    st.write(f"입찰참가자격등록마감일시: {row.get('bidQlfctRgstDt', '-')}")

    officer = row.get("ntceInsttOfclNm") or row.get("exctvNm")
    if officer:
        st.write(f"담당자명: {officer}")
    phone = row.get("ntceInsttOfclTelNo")
    if phone:
        phone_str = str(phone)
        if "*" in phone_str:
            st.write("담당자 전화번호: 비공개")
        else:
            st.write(f"담당자 전화번호: {phone_str}")

    st.write("---")
    st.write("공고규격서 및 첨부파일")
    files = []
    for i in range(1, 11):
        url = row.get(f"ntceSpecDocUrl{i}")
        name = row.get(f"ntceSpecFileNm{i}")
        if pd.notna(url) and isinstance(url, str) and url.strip():
            if not name or not isinstance(name, str) or not name.strip():
                name = f"첨부파일 {i}"
            files.append((name, url))
    if files:
        for idx, (name, url) in enumerate(files, start=1):
            st.markdown(f"{idx}. [{name}]({url})")
    else:
        st.write("첨부파일이 없습니다.")

    if st.button("닫기"):
        st.query_params.clear()
        st.rerun()


def main():
    df, data_source = load_data()

    if df.empty:
        if not FIREBASE_ENABLED:
            st.error("Firebase 설정을 찾을 수 없습니다. secrets 또는 JSON 경로를 확인해 주세요.")
        elif data_source.startswith("error:"):
            st.error(f"Firebase에서 데이터를 불러오는 중 오류가 발생했습니다:\n{data_source[7:]}")
        else:
            st.info("Firestore에 표시할 입찰공고 데이터가 없습니다.")
        return
    flash_message = st.session_state.pop("refresh_message", None)
    if flash_message:
        level = flash_message.get("type", "info")
        text = flash_message.get("text", "")
        icon_map = {"success": "✅", "error": "❌", "info": "ℹ️"}
        icon = icon_map.get(level, "ℹ️")
        if text:
            st.toast(text, icon=icon)
    latest_bid_dt = get_latest_bid_datetime()
    last_collection_date = get_last_collection_date()

    # ===== 상단 헤더 =====
    st.title("나라장터 용역 입찰공고 모니터링")

    with st.container():
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            if last_collection_date:
                st.caption(f"마지막 업데이트: {last_collection_date.isoformat()}")
            elif latest_bid_dt:
                st.caption(f"마지막 업데이트: {latest_bid_dt.strftime('%Y-%m-%d')}")
        with col_h2:
            if st.button("최신 데이터 불러오기", use_container_width=True):
                with st.spinner("최신 데이터를 수집하고 있습니다..."):
                    last_date = last_collection_date or (
                        latest_bid_dt.date() if latest_bid_dt else None
                    )
                    today = now_kst().date()
                    if last_date == today:
                        st.session_state["refresh_message"] = {
                            "type": "info",
                            "text": f"현재 최신 데이터입니다. 마지막 업데이트 날짜: {last_date.strftime('%Y-%m-%d')}",
                        }
                        st.rerun()

                    try:
                        result = collect_and_upsert(verbose=False)
                    except Exception as exc:
                        st.session_state["refresh_message"] = {
                            "type": "error",
                            "text": f"데이터 수집에 실패했습니다: {exc}",
                        }
                    else:
                        if result["upserted_records"] == 0:
                            msg_date = (
                                last_date.strftime("%Y-%m-%d")
                                if last_date
                                else "-"
                            )
                            st.session_state["refresh_message"] = {
                                "type": "info",
                                "text": f"현재 최신 데이터입니다. 마지막 업데이트 날짜: {msg_date}",
                            }
                        else:
                            start_str = result["start_dt"].strftime("%Y-%m-%d %H:%M")
                            end_str = result["end_dt"].strftime("%Y-%m-%d %H:%M")
                            st.session_state["refresh_message"] = {
                                "type": "success",
                                "text": f"{start_str} ~ {end_str} 구간에서 {result['upserted_records']}건 업데이트 완료",
                            }
                        st.cache_data.clear()
                    st.rerun()
            st.write("")

    st.markdown("---")

    # ===== 필터 영역 (사이드바) =====
    st.sidebar.header("검색 조건")

    # 조회 기준 (inqryDiv 개념)
    inqry_div = st.sidebar.radio(
        "조회 기준",
        options=["등록일시 기준", "입찰공고번호"],
        index=0,
    )

    # 기간/공고번호
    today = now_kst().date()
    default_start = today - timedelta(days=90)

    if inqry_div == "등록일시 기준":
        quick_range = st.sidebar.selectbox(
            "빠른 기간 선택",
            options=["직접 선택", "최근 1주", "최근 1개월", "최근 3개월"],
            index=3,
        )

        if quick_range == "최근 1주":
            start_date = today - timedelta(days=7)
            end_date = today
        elif quick_range == "최근 1개월":
            start_date = today - timedelta(days=30)
            end_date = today
        elif quick_range == "최근 3개월":
            start_date = default_start
            end_date = today
        else:
            start_date = st.sidebar.date_input(
                "시작일",
                value=default_start,
            )
            end_date = st.sidebar.date_input(
                "종료일",
                value=today,
            )

        # df 필터용
        target_col = "bidNtceDt"
    else:
        bid_no_query = st.sidebar.text_input("입찰공고번호", value="")

    # 공고명 검색
    # 예가 필터
    st.sidebar.subheader("예가 필터")
    min_budget_toggle = st.sidebar.checkbox("배정예산 20억 이상만 보기", value=False)
    if min_budget_toggle:
        min_budget = 2_000_000_000
    else:
        min_budget = st.sidebar.number_input(
            "최소 배정예산 (원)",
            min_value=0,
            value=0,
            step=100_000_000,
        )

    st.sidebar.markdown("**공고명 검색**")
    bid_title_query = st.sidebar.text_input(
        "공고명 검색",
        value="",
        label_visibility="collapsed",
        placeholder="공고명을 입력하세요",
    )

    # 기관 필터
    st.sidebar.subheader("기관 필터")
    inst_keyword = st.sidebar.text_input("공고기관명 / 수요기관명", value="")

    # 재공고 여부
    st.sidebar.subheader("재공고 여부")
    reopt = st.sidebar.selectbox(
        "재공고 필터",
        options=["전체", "Y", "N"],
        index=0,
    )

    # 상태 필터
    st.sidebar.subheader("입찰 상태")
    status_opt = st.sidebar.selectbox(
        "입찰 상태 필터",
        options=["전체", "입찰 전", "진행중", "마감", "정보 부족"],
        index=0,
    )

    st.sidebar.markdown("---")
    apply_filter = st.sidebar.button("검색 적용")

    # ===== 필터 적용 로직 =====
    filtered = df.copy()

    # 날짜 / 공고번호 기반 필터
    if inqry_div == "등록일시 기준":
        if isinstance(start_date, datetime):
            start_dt = start_date
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())

        if isinstance(end_date, datetime):
            end_dt = end_date
        else:
            end_dt = datetime.combine(end_date, datetime.max.time())

        if target_col in filtered.columns:
            filtered = filtered[
                (filtered[target_col] >= start_dt) & (filtered[target_col] <= end_dt)
            ]
    else:
        if bid_no_query:
            filtered = filtered[filtered["bidNtceNo"].astype(str).str.contains(bid_no_query.strip())]

    # 예가 필터 (배정예산금액 기준)
    if "asignBdgtAmt" in filtered.columns:
        filtered = filtered[filtered["asignBdgtAmt"] >= min_budget]

    # 공고명 검색 필터
    if bid_title_query:
        pattern = bid_title_query.strip()
        filtered = filtered[filtered["bidNtceNm"].str.contains(pattern, case=False, na=False)]

    # 기관 필터
    if inst_keyword:
        inst_pattern = inst_keyword.strip()
        filtered = filtered[
            filtered["ntceInsttNm"].str.contains(inst_pattern, case=False, na=False)
            | filtered["dminsttNm"].str.contains(inst_pattern, case=False, na=False)
        ]

    # 재공고 필터
    if reopt in {"Y", "N"}:
        filtered = filtered[filtered["reNtceYn"] == reopt]

    # 상태 필터
    if status_opt != "전체":
        filtered = filtered[filtered["status"] == status_opt]

    # ===== 입찰공고 목록 테이블 =====
    total_count = len(filtered)
    avg_budget = (
        filtered["asignBdgtAmt"].mean()
        if "asignBdgtAmt" in filtered.columns and total_count > 0
        else None
    )
    max_budget = (
        filtered["asignBdgtAmt"].max()
        if "asignBdgtAmt" in filtered.columns and total_count > 0
        else None
    )
    re_count = (
        (filtered["reNtceYn"] == "Y").sum()
        if "reNtceYn" in filtered.columns and total_count > 0
        else 0
    )
    re_ratio = (re_count / total_count * 100) if total_count > 0 else None

    st.markdown("### 요약 지표")
    card_cols = st.columns(4)
    with card_cols[0]:
        st.metric("검색 결과 공고 수", f"{total_count:,}건")
    with card_cols[1]:
        st.metric("평균 배정예산", format_money(avg_budget) if avg_budget is not None else "-")
    with card_cols[2]:
        st.metric("최대 배정예산", format_money(max_budget) if max_budget is not None else "-")
    with card_cols[3]:
        if re_ratio is not None:
            st.metric("재공고 비율", f"{re_ratio:.0f}% ({re_count}건)")
        else:
            st.metric("재공고 비율", "-")

    st.markdown("---")
    st.subheader("입찰공고 목록")

    if total_count == 0:
        st.info("조건에 해당하는 공고가 없습니다. 필터를 조정해보세요.")
        return

    df_disp = filtered.copy()

    now = now_kst_naive()
    df_disp["기관명"] = df_disp["dminsttNm"].fillna("").replace("", pd.NA)
    df_disp["기관명"] = df_disp["기관명"].fillna(df_disp["ntceInsttNm"])

    # 문자열 포맷
    def fmt_dt(series_name):
        series = filtered[series_name] if series_name in filtered.columns else None
        if series is not None and pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.strftime("%Y-%m-%d %H:%M")
        return df_disp.get(series_name, "-")

    df_disp["입찰공고일시"] = fmt_dt("bidNtceDt")
    df_disp["입찰참가자격등록마감일시"] = fmt_dt("bidQlfctRgstDt")

    if "bidClseDt" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["bidClseDt"]):
        df_disp["상태"] = filtered["bidClseDt"].apply(
            lambda x: "-"
            if pd.isna(x)
            else ("마감" if x < now else "진행중")
        )
        df_disp["마감까지 남은시간"] = filtered["bidClseDt"].apply(
            lambda x: "-"
            if pd.isna(x)
            else ("마감" if x < now else f"{(x - now).days}일 {(x - now).seconds // 3600}시간 남음")
        )
    else:
        df_disp["상태"] = "-"
        df_disp["마감까지 남은시간"] = "-"

    df_disp["배정예산금액표시"] = df_disp["asignBdgtAmt"].apply(format_money) if "asignBdgtAmt" in df_disp else "-"
    df_disp["추정가격표시"] = df_disp["presmptPrce"].apply(format_money) if "presmptPrce" in df_disp else "-"
    df_disp["재공고표시"] = df_disp["reNtceYn"].fillna("-") if "reNtceYn" in df_disp else "-"

    df_disp["_source_index"] = filtered.index
    df_disp = df_disp.sort_values(
        by=["재공고표시", "상태", "입찰공고일시"],
        ascending=[False, False, False],
    )

    column_headers = [
        "입찰공고일시",
        "공고명",
        "배정예산금액",
        "추정가격",
        "기관명",
        "상태",
        "재공고 여부",
    ]
    column_widths = [1.3, 2.8, 1.3, 1.3, 1.6, 1.1, 1.1]

    header_cols = st.columns(column_widths, gap="small")
    for col, title in zip(header_cols, column_headers):
        col.markdown(f"<div style='text-align:center;font-weight:bold'>{title}</div>", unsafe_allow_html=True)

    for _, row in df_disp.iterrows():
        src_row = filtered.loc[row["_source_index"]]
        row_cols = st.columns(column_widths, gap="small")

        row_cols[0].markdown(f"<div style='text-align:center'>{row.get('입찰공고일시') or '-'}</div>", unsafe_allow_html=True)

        full_title = row.get("bidNtceNm") or "-"
        short_title = full_title if len(full_title) <= 36 else f"{full_title[:33]}..."
        with row_cols[1]:
            if st.button(
                short_title,
                key=f"detail_{src_row.get('bidNtceNo')}_{src_row.get('bidNtceOrd', '')}",
                help=full_title,
                use_container_width=True,
                type="secondary",
            ):
                show_detail_dialog(src_row)

        row_cols[2].markdown(f"<div style='text-align:center'>{row.get('배정예산금액표시') or '-'}</div>", unsafe_allow_html=True)
        row_cols[3].markdown(f"<div style='text-align:center'>{row.get('추정가격표시') or '-'}</div>", unsafe_allow_html=True)
        row_cols[4].markdown(f"<div style='text-align:center'>{row.get('기관명') or '-'}</div>", unsafe_allow_html=True)
        row_cols[5].markdown(f"<div style='text-align:center'>{row.get('상태') or '-'}</div>", unsafe_allow_html=True)
        re_value = row.get("재공고표시") or "-"
        re_display = re_value if re_value in ("Y", "N") else "-"
        row_cols[6].markdown(f"<div style='text-align:center'>{re_display}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
