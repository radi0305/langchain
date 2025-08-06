#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
표준 한의학 용어집 관리자 - medical_terms_manager.py (데이터 디렉토리 분리 버전)
대한한의학회 표준한의학용어집 기반 지능형 쿼리 확장 시스템
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")


class MedicalTermsManager:
    """표준 한의학 용어집 기반 관리자"""

    def __init__(self,
                 terms_file: str = "/Users/radi/Projects/langchain/hmedicalterms.json",
                 cache_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG/cache"):
        """표준용어집 관리자 초기화"""
        self.terms_file = Path(terms_file)
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # 캐시 파일 경로
        self.cache_file = self.cache_path / 'medical_terms_index.pkl'

        # 인덱스 초기화
        self.terms_data = []
        self.search_index = {}
        self.category_index = {}
        self.synonym_index = {}
        self.hierarchical_index = {}

        # 로딩 시도
        self._load_or_build_index()

    def _load_or_build_index(self):
        """인덱스 로드 또는 생성"""
        try:
            if self._should_rebuild_cache():
                print("📚 표준용어집 인덱스 생성 중...")
                self._build_index()
                self._save_cache()
                print("✅ 표준용어집 인덱스 생성 완료")
            else:
                print("📚 표준용어집 캐시에서 로딩 중...")
                self._load_cache()
                print("✅ 표준용어집 캐시 로드 완료")
        except Exception as e:
            print(f"⚠️ 표준용어집 처리 중 오류: {e}")
            print("📚 기본 인덱스로 초기화합니다.")
            self._initialize_basic_index()

    def _should_rebuild_cache(self) -> bool:
        """캐시 재생성 필요 여부 확인"""
        if not self.cache_file.exists():
            return True

        if not self.terms_file.exists():
            print(f"⚠️ 표준용어집 파일을 찾을 수 없습니다: {self.terms_file}")
            return False

        # 파일 수정 시간 비교
        try:
            cache_mtime = self.cache_file.stat().st_mtime
            terms_mtime = self.terms_file.stat().st_mtime
            return terms_mtime > cache_mtime
        except Exception:
            return True

    def _build_index(self):
        """표준용어집 인덱스 생성"""
        if not self.terms_file.exists():
            print(f"⚠️ 표준용어집 파일이 없습니다: {self.terms_file}")
            self._initialize_basic_index()
            return

        try:
            with open(self.terms_file, 'r', encoding='utf-8') as f:
                self.terms_data = json.load(f)

            print(f"📊 {len(self.terms_data)}개 용어 로드 완료")

            # 각종 인덱스 구축
            self._build_search_index()
            self._build_category_index()
            self._build_synonym_index()
            self._build_hierarchical_index()

        except Exception as e:
            print(f"⚠️ 표준용어집 파싱 실패: {e}")
            self._initialize_basic_index()

    def _build_search_index(self):
        """검색 인덱스 구축"""
        self.search_index = {}

        for term_data in self.terms_data:
            # 기본 용어명
            term_name = term_data.get('용어명', '')
            if term_name:
                self.search_index[term_name] = term_data

            # 한자명
            hanja_name = term_data.get('용어명_한자', '')
            if hanja_name and hanja_name != term_name:
                self.search_index[hanja_name] = term_data

            # 동의어
            synonyms = term_data.get('동의어', [])
            for synonym in synonyms:
                if synonym:
                    self.search_index[synonym] = term_data

            # 검색키워드
            keywords = term_data.get('검색키워드', [])
            for keyword in keywords:
                if keyword and keyword not in self.search_index:
                    self.search_index[keyword] = term_data

    def _build_category_index(self):
        """분류별 인덱스 구축"""
        self.category_index = defaultdict(list)

        for term_data in self.terms_data:
            category = term_data.get('분류', '기타')
            self.category_index[category].append(term_data)

    def _build_synonym_index(self):
        """동의어 인덱스 구축"""
        self.synonym_index = {}

        for term_data in self.terms_data:
            term_name = term_data.get('용어명', '')
            synonyms = term_data.get('동의어', [])

            if term_name and synonyms:
                all_terms = [term_name] + synonyms
                # 각 용어에 대해 다른 모든 용어를 동의어로 등록
                for term in all_terms:
                    if term:
                        related = [t for t in all_terms if t != term]
                        self.synonym_index[term] = related

    def _build_hierarchical_index(self):
        """계층구조 인덱스 구축"""
        self.hierarchical_index = {}

        for term_data in self.terms_data:
            term_name = term_data.get('용어명', '')
            hierarchy = term_data.get('계층구조', {})

            if term_name and hierarchy:
                self.hierarchical_index[term_name] = {
                    'parents': hierarchy.get('상위개념', []),
                    'children': hierarchy.get('하위개념', [])
                }

    def _save_cache(self):
        """캐시 저장"""
        try:
            cache_data = {
                'terms_data': self.terms_data,
                'search_index': self.search_index,
                'category_index': dict(self.category_index),
                'synonym_index': self.synonym_index,
                'hierarchical_index': self.hierarchical_index,
                'timestamp': self._get_current_timestamp()
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")

    def _load_cache(self):
        """캐시 로드"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.terms_data = cache_data.get('terms_data', [])
            self.search_index = cache_data.get('search_index', {})
            self.category_index = cache_data.get('category_index', {})
            self.synonym_index = cache_data.get('synonym_index', {})
            self.hierarchical_index = cache_data.get('hierarchical_index', {})

        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            self._initialize_basic_index()

    def _initialize_basic_index(self):
        """기본 인덱스 초기화 (폴백)"""
        print("📚 기본 중의학 용어로 초기화합니다.")

        basic_terms = [
            {'용어명': '혈허', '용어명_한자': '血虛', '분류': '병증', '동의어': [
                '혈부족'], '검색키워드': ['혈허', '혈부족', '血虛']},
            {'용어명': '기허', '용어명_한자': '氣虛', '분류': '병증', '동의어': [
                '기부족'], '검색키워드': ['기허', '기부족', '氣虛']},
            {'용어명': '음허', '용어명_한자': '陰虛', '분류': '병증', '동의어': [
                '음부족'], '검색키워드': ['음허', '음부족', '陰虛']},
            {'용어명': '양허', '용어명_한자': '陽虛', '분류': '병증', '동의어': [
                '양부족'], '검색키워드': ['양허', '양부족', '陽虛']},
            {'용어명': '사물탕', '용어명_한자': '四物湯', '분류': '처방',
                '동의어': [], '검색키워드': ['사물탕', '四物湯']},
            {'용어명': '사군자탕', '용어명_한자': '四君子湯', '분류': '처방',
                '동의어': [], '검색키워드': ['사군자탕', '四君子湯']},
            {'용어명': '육군자탕', '용어명_한자': '六君子湯', '분류': '처방',
                '동의어': [], '검색키워드': ['육군자탕', '六君子湯']},
            {'용어명': '보중익기탕', '용어명_한자': '補中益氣湯', '분류': '처방',
                '동의어': [], '검색키워드': ['보중익기탕', '補中益氣湯']},
            {'용어명': '당귀보혈탕', '용어명_한자': '當歸補血湯', '분류': '처방',
                '동의어': [], '검색키워드': ['당귀보혈탕', '當歸補血湯']},
            {'용어명': '인삼', '용어명_한자': '人參', '분류': '약물', '동의어': [
                '고려삼', '홍삼'], '검색키워드': ['인삼', '人參', '고려삼', '홍삼']},
            {'용어명': '당귀', '용어명_한자': '當歸', '분류': '약물',
                '동의어': [], '검색키워드': ['당귀', '當歸']},
            {'용어명': '천궁', '용어명_한자': '川芎', '분류': '약물',
                '동의어': [], '검색키워드': ['천궁', '川芎']},
            {'용어명': '백작약', '용어명_한자': '白芍藥', '분류': '약물',
                '동의어': ['백작'], '검색키워드': ['백작약', '白芍藥', '백작']},
            {'용어명': '숙지황', '용어명_한자': '熟地黃', '분류': '약물',
                '동의어': [], '검색키워드': ['숙지황', '熟地黃']},
            {'용어명': '황기', '용어명_한자': '黃芪', '분류': '약물',
                '동의어': [], '검색키워드': ['황기', '黃芪']},
            {'용어명': '백출', '용어명_한자': '白朮', '분류': '약물',
                '동의어': [], '검색키워드': ['백출', '白朮']},
            {'용어명': '복령', '용어명_한자': '茯苓', '분류': '약물',
                '동의어': [], '검색키워드': ['복령', '茯苓']},
            {'용어명': '감초', '용어명_한자': '甘草', '분류': '약물',
                '동의어': [], '검색키워드': ['감초', '甘草']},
        ]

        self.terms_data = basic_terms
        self._build_search_index()
        self._build_category_index()
        self._build_synonym_index()
        self._build_hierarchical_index()

    def _get_current_timestamp(self):
        """현재 타임스탬프 반환"""
        return datetime.now().isoformat()

    def expand_query(self, query: str, max_expansions: int = 10) -> List[str]:
        """
        지능형 쿼리 확장
        6단계 확장 전략: 직접매칭 → 동의어 → 부분매칭 → 계층구조 → 카테고리 → 기본매핑
        """
        expansions = set([query])  # 중복 방지를 위해 set 사용

        try:
            # 1단계: 직접 매칭
            if query in self.search_index:
                term_data = self.search_index[query]

                # 동의어 추가
                synonyms = term_data.get('동의어', [])
                for synonym in synonyms[:3]:  # 상위 3개만
                    if synonym:
                        expansions.add(synonym)

                # 한자/한글명 상호 추가
                hanja = term_data.get('용어명_한자', '')
                hangul = term_data.get('용어명', '')

                if hanja and hanja != query:
                    expansions.add(hanja)
                if hangul and hangul != query:
                    expansions.add(hangul)

                # 검색키워드 추가
                keywords = term_data.get('검색키워드', [])
                for keyword in keywords[:2]:  # 상위 2개만
                    if keyword and keyword != query:
                        expansions.add(keyword)

            # 2단계: 동의어 기반 확장
            if query in self.synonym_index:
                related_terms = self.synonym_index[query]
                for term in related_terms[:2]:  # 상위 2개만
                    if term:
                        expansions.add(term)

            # 3단계: 부분 매칭 (포함 관계)
            partial_matches = []
            for term in self.search_index.keys():
                if (query in term or term in query) and term != query:
                    # 길이 차이가 너무 크지 않은 것만
                    if abs(len(term) - len(query)) <= 3:
                        partial_matches.append(term)

            # 부분 매칭 결과를 관련성으로 정렬
            partial_matches.sort(key=lambda x: abs(len(x) - len(query)))
            for match in partial_matches[:3]:  # 상위 3개만
                expansions.add(match)

            # 4단계: 계층구조 기반 확장
            if query in self.hierarchical_index:
                hierarchy = self.hierarchical_index[query]

                # 상위개념 추가
                parents = hierarchy.get('parents', [])
                for parent in parents[:2]:  # 상위 2개만
                    if parent:
                        expansions.add(parent)

                # 하위개념 추가 (1개만)
                children = hierarchy.get('children', [])
                if children:
                    expansions.add(children[0])

            # 5단계: 카테고리 기반 확장
            query_category = None
            if query in self.search_index:
                query_category = self.search_index[query].get('분류')

                if query_category and query_category in self.category_index:
                    category_terms = self.category_index[query_category]

                    # 같은 카테고리의 다른 용어 추가 (2개만)
                    added_from_category = 0
                    for term_data in category_terms:
                        if added_from_category >= 2:
                            break

                        term_name = term_data.get('용어명', '')
                        if term_name and term_name != query and term_name not in expansions:
                            expansions.add(term_name)
                            added_from_category += 1

            # 6단계: 기본 매핑 (한의학 도메인 지식)
            basic_expansions = self._get_basic_domain_expansions(query)
            for basic_exp in basic_expansions[:2]:  # 상위 2개만
                expansions.add(basic_exp)

            # 리스트로 변환하고 길이 제한
            result_list = list(expansions)

            # 품질 필터링 (2글자 이상, 의미있는 용어)
            filtered_result = []
            for term in result_list:
                if len(term) >= 2 and self._is_meaningful_term(term):
                    filtered_result.append(term)

            # 길이 제한
            return filtered_result[:max_expansions]

        except Exception as e:
            print(f"⚠️ 쿼리 확장 실패: {e}")
            return [query]

    def _get_basic_domain_expansions(self, query: str) -> List[str]:
        """기본 도메인 지식 기반 확장"""
        basic_mappings = {
            # 허증 관련
            '血虛': ['補血', '當歸', '四物湯', '陰血不足'],
            '氣虛': ['補氣', '人參', '黃芪', '四君子湯'],
            '陰虛': ['滋陰', '補陰', '六味地黃湯', '生地黃'],
            '陽虛': ['溫陽', '補陽', '腎氣丸', '附子'],

            # 처방 관련
            '四物湯': ['當歸', '川芎', '白芍', '熟地黃', '補血'],
            '四君子湯': ['人參', '白朮', '茯苓', '甘草', '補氣'],
            '六君子湯': ['四君子湯', '陳皮', '半夏', '化痰'],
            '補中益氣湯': ['黃芪', '人參', '當歸', '升麻', '補氣升陽'],

            # 약물 관련
            '人參': ['補氣', '大補元氣', '復脈', '生津'],
            '當歸': ['補血', '活血', '調經', '四物湯'],
            '黃芪': ['補氣', '升陽', '固表', '利水'],

            # 증상 관련
            '失眠': ['不寐', '心神不安', '養心安神'],
            '眩暈': ['頭暈', '肝陽上亢', '痰濁中阻'],
            '心悸': ['驚悸', '心神不寧', '心氣不足'],
        }

        return basic_mappings.get(query, [])

    def _is_meaningful_term(self, term: str) -> bool:
        """의미있는 용어인지 판단"""
        if len(term) < 2:
            return False

        # 단순 반복 문자 제외
        if len(set(term)) == 1:
            return False

        # 특수문자만 있는 경우 제외
        if not any('\u4e00' <= char <= '\u9fff' or char.isalpha() for char in term):
            return False

        return True

    def split_query_intelligently(self, query: str) -> List[str]:
        """지능적 쿼리 분할"""
        parts = [query]

        try:
            # 길이별 분할 전략
            if len(query) >= 4:
                # 4글자 이상: 2글자씩 분할 + 3글자씩 분할
                for i in range(len(query) - 1):
                    if i + 2 <= len(query):
                        part = query[i:i + 2]
                        parts.append(part)

                for i in range(len(query) - 2):
                    if i + 3 <= len(query):
                        part = query[i:i + 3]
                        parts.append(part)

            elif len(query) == 3:
                # 3글자: 앞 2글자, 뒤 2글자 추가
                parts.append(query[:2])
                parts.append(query[1:])

            # 중복 제거 및 의미있는 부분만 선택
            unique_parts = []
            for part in parts:
                if part not in unique_parts and self._is_meaningful_term(part):
                    unique_parts.append(part)

            return unique_parts

        except Exception as e:
            print(f"⚠️ 쿼리 분할 실패: {e}")
            return [query]

    def get_related_terms(self, query: str) -> List[str]:
        """관련 용어 반환"""
        related = []

        try:
            if query in self.search_index:
                term_data = self.search_index[query]

                # 동의어 추가
                synonyms = term_data.get('동의어', [])
                related.extend(synonyms[:3])

                # 같은 분류의 다른 용어들
                category = term_data.get('분류', '')
                if category in self.category_index:
                    category_terms = self.category_index[category]
                    for cat_term in category_terms[:5]:
                        term_name = cat_term.get('용어명', '')
                        if term_name and term_name != query and term_name not in related:
                            related.append(term_name)

                # 계층구조 관련 용어
                if query in self.hierarchical_index:
                    hierarchy = self.hierarchical_index[query]
                    parents = hierarchy.get('parents', [])
                    children = hierarchy.get('children', [])

                    related.extend(parents[:2])
                    related.extend(children[:2])

            # 기본 도메인 확장도 추가
            domain_related = self._get_basic_domain_expansions(query)
            related.extend(domain_related[:3])

            # 중복 제거 및 길이 제한
            unique_related = []
            for term in related:
                if term and term not in unique_related and term != query:
                    unique_related.append(term)

            return unique_related[:10]

        except Exception as e:
            print(f"⚠️ 관련 용어 추출 실패: {e}")
            return []

    def get_term_info(self, term: str) -> Optional[Dict]:
        """특정 용어의 상세 정보 반환"""
        try:
            if term in self.search_index:
                return self.search_index[term]
            return None
        except Exception as e:
            print(f"⚠️ 용어 정보 조회 실패: {e}")
            return None

    def search_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """분류별 용어 검색"""
        try:
            if category in self.category_index:
                terms = self.category_index[category]
                return terms[:limit]
            return []
        except Exception as e:
            print(f"⚠️ 카테고리 검색 실패: {e}")
            return []

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[str]:
        """유사 용어 검색"""
        matches = []

        try:
            for term in self.search_index.keys():
                if term == query:
                    continue

                # 간단한 유사도 계산 (문자 겹침 비율)
                common_chars = set(query) & set(term)
                similarity = len(common_chars) / max(len(query), len(term))

                if similarity >= threshold:
                    matches.append((term, similarity))

            # 유사도 순으로 정렬
            matches.sort(key=lambda x: x[1], reverse=True)
            return [match[0] for match in matches[:10]]

        except Exception as e:
            print(f"⚠️ 유사 검색 실패: {e}")
            return []

    def get_statistics(self) -> Dict:
        """용어집 통계 정보 반환"""
        try:
            stats = {
                'total_terms': len(self.terms_data),
                'search_index_size': len(self.search_index),
                'categories': len(self.category_index),
                'terms_with_synonyms': len(self.synonym_index),
                'terms_with_hierarchy': len(self.hierarchical_index)
            }

            # 카테고리별 분포
            category_distribution = {}
            for category, terms in self.category_index.items():
                category_distribution[category] = len(terms)

            stats['category_distribution'] = category_distribution
            return stats

        except Exception as e:
            print(f"⚠️ 통계 정보 생성 실패: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """캐시 파일 삭제"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                print("🗑️ 표준용어집 캐시가 삭제되었습니다.")
            else:
                print("💭 삭제할 캐시 파일이 없습니다.")
        except Exception as e:
            print(f"⚠️ 캐시 삭제 실패: {e}")

    def rebuild_index(self):
        """강제로 인덱스 재구축"""
        try:
            print("🔄 표준용어집 인덱스 강제 재구축 중...")
            self._build_index()
            self._save_cache()
            print("✅ 인덱스 재구축 완료")
        except Exception as e:
            print(f"⚠️ 인덱스 재구축 실패: {e}")


# 편의 함수들
def create_terms_manager() -> MedicalTermsManager:
    """표준용어집 관리자 생성 편의 함수"""
    return MedicalTermsManager()


def test_terms_manager():
    """테스트용 함수"""
    print("🧪 표준용어집 관리자 테스트")

    manager = MedicalTermsManager()

    # 통계 정보 출력
    stats = manager.get_statistics()
    print(f"📊 총 용어 수: {stats.get('total_terms', 0):,}개")
    print(f"🔍 검색 인덱스: {stats.get('search_index_size', 0):,}개")
    print(f"🏷️ 카테고리: {stats.get('categories', 0)}개")

    # 쿼리 확장 테스트
    test_queries = ['血虛', '四君子湯', '人參', '陰虛']

    for query in test_queries:
        print(f"\n🔍 '{query}' 쿼리 확장 테스트:")
        expansions = manager.expand_query(query)
        print(f"   확장 결과: {expansions}")

        related = manager.get_related_terms(query)
        if related:
            print(f"   관련 용어: {related[:5]}")


if __name__ == "__main__":
    test_terms_manager()
