
# 나라장터 용역 입찰공고 모니터링 대시보드

- 내부 사업 기획용 실시간 용역 입찰공고 모니터링 Streamlit 대시보드
- Firestore 적재 공고 메타데이터 기반 필터링·정렬·상세 열람 흐름
- KST 기준 증분 업서트를 수행하는 G2B 수집 스크립트
- 운영 인스턴스: `https://g2b-bid-finder-wvd372mhbn8kmegqeafk9y.streamlit.app/`

## 주요 기능

- **데이터 수집 자동화**: G2B API 구간별 호출 및 Firestore 업서트를 수행하는 `export_ax_oct.py` 파이프라인
- **상세 분석 UI**: 공고 상태·재공고 여부·예산 지표·마감까지 남은 시간 확인을 위한 요약 카드와 리스트/다이얼로그 구성
- **고급 필터링**: 등록일시/공고번호 기준 조회, 예산 하한, 공고명 키워드, 기관명, 재공고 여부, 상태 조건 제공
- **첨부파일 접근성**: 공고별 규격서·첨부파일 링크 및 담당자 연락처 통합 노출

## 시스템 개요

- **수집 파이프라인**: KST 기준 마지막 `bidNtceDt` 이후 구간 증분 요청과 Firestore `bid_pblanc_list`·`meta.collection_state` 갱신 흐름
- **대시보드**: Firestore 데이터 캐싱 조회 후 Streamlit 컴포넌트로 목록·상세 뷰를 제공하는 `app.py`
- **주요 스키마**: 공고 메타데이터(`bidNtceNm`, `asignBdgtAmt`, `presmptPrce`, `reNtceYn` 등), 일정 정보(`bidNtceDt`, `bidBeginDt`, `bidClseDt`, `bidQlfctRgstDt`), 담당자/첨부파일, 식별자(`untyNtceNo`, `bidNtceNo`)

## 운영 가이드

- 서비스 키 및 Firebase 자격 증명 비공개 저장, Streamlit secrets 활용 권장

## English Summary

- Streamlit dashboard for the business planning and sales team to monitor service bid notices collected via the `getBidPblancListInfoServc` API
- Filtering, sorting, and detailed inspection workflows powered by Firestore-stored bid metadata
- G2B harvesting script that performs KST-based incremental upserts
- Production instance: `https://g2b-bid-finder-wvd372mhbn8kmegqeafk9y.streamlit.app/`

### Key Features

- **Automated harvesting**: `export_ax_oct.py` pipeline that batches G2B API calls and upserts results into Firestore
- **Analytical UI**: summary cards and list/dialog views exposing bid status, re-announcement flags, budget metrics, and remaining time
- **Advanced filters**: query options for registration datetime, notice ID, budget threshold, title keyword, organization, re-announcement flag, and status
- **Attachment access**: consolidated display of specification files and stakeholder contact information per notice

### Architecture Overview

- **Ingestion pipeline**: incremental requests after the latest `bidNtceDt` in KST and updates to Firestore `bid_pblanc_list` / `meta.collection_state`
- **Dashboard layer**: `app.py` retrieving and caching Firestore data before rendering Streamlit list/detail components
- **Core schema**: bid metadata (`bidNtceNm`, `asignBdgtAmt`, `presmptPrce`, `reNtceYn`, etc.), schedule fields (`bidNtceDt`, `bidBeginDt`, `bidClseDt`, `bidQlfctRgstDt`), contacts/attachments, identifiers (`untyNtceNo`, `bidNtceNo`)

### Operations Guidance

- Secure storage of service keys and Firebase credentials with Streamlit secrets recommended

