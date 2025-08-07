#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
표준 한의학 용어집 관리자 - medical_terms_manager_improved.py (개선된 버전)
하드코딩된 기본 매핑을 강화하고 관계 그래프 기반 확장 시스템 구축
대한한의학회 표준한의학용어집 기반 지능형 쿼리 확장 시스템
"""

import json
import pickle
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")


class MedicalTermsManager:
    """표준 한의학 용어집 기반 관리자 (개선된 버전)"""

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

        # 개선된 기능들
        self.relationship_graph = nx.Graph()
        self.semantic_clusters = {}
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
        self.expansion_patterns = {}
        self.domain_knowledge_base = {}

        # 로딩 시도
        self._load_or_build_index()

        # 고급 분석 수행
        self._build_relationship_graph()
        self._analyze_semantic_clusters()
        self._build_domain_knowledge_base()

    def _load_or_build_index(self):
        """인덱스 로드 또는 생성"""
        try:
            if self._should_rebuild_cache():
                print("📚 표준용어집 고급 인덱스 생성 중...")
                self._build_index()
                self._save_cache()
                print("✅ 표준용어집 고급 인덱스 생성 완료")
            else:
                print("📚 표준용어집 캐시에서 로딩 중...")
                self._load_cache()
                print("✅ 표준용어집 캐시 로드 완료")
        except Exception as e:
            print(f"⚠️ 표준용어집 처리 중 오류: {e}")
            print("📚 고급 기본 인덱스로 초기화합니다.")
            self._initialize_advanced_basic_index()

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
            self._initialize_advanced_basic_index()
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

            # 추가 인덱스 구축
            self._build_co_occurrence_matrix()
            self._extract_expansion_patterns()

        except Exception as e:
            print(f"⚠️ 표준용어집 파싱 실패: {e}")
            self._initialize_advanced_basic_index()

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

    def _build_co_occurrence_matrix(self):
        """공기 행렬 구축 (용어 간 관련성 분석)"""
        print("🔗 용어 간 공기 관계 분석 중...")

        for term_data in self.terms_data:
            term_name = term_data.get('용어명', '')
            category = term_data.get('분류', '')
            keywords = term_data.get('검색키워드', [])
            hierarchy = term_data.get('계층구조', {})

            # 같은 카테고리 용어들 간의 관련성
            if category:
                category_terms = [t.get('용어명', '')
                                  for t in self.category_index[category]]
                for related_term in category_terms:
                    if related_term and related_term != term_name:
                        self.co_occurrence_matrix[term_name][related_term] += 1

            # 키워드 기반 관련성
            for keyword in keywords:
                if keyword and keyword != term_name:
                    self.co_occurrence_matrix[term_name][keyword] += 2

            # 계층구조 기반 관련성
            parents = hierarchy.get('상위개념', [])
            children = hierarchy.get('하위개념', [])

            for parent in parents:
                if parent:
                    self.co_occurrence_matrix[term_name][parent] += 3

            for child in children:
                if child:
                    self.co_occurrence_matrix[term_name][child] += 3

    def _extract_expansion_patterns(self):
        """확장 패턴 추출"""
        print("🔍 확장 패턴 분석 중...")

        # 카테고리별 패턴 분석
        category_patterns = defaultdict(list)

        for category, terms in self.category_index.items():
            if len(terms) < 5:  # 최소 5개 이상의 용어가 있는 카테고리만
                continue

            # 공통 패턴 추출
            common_chars = self._find_common_patterns(
                [t.get('용어명_한자', '') for t in terms])
            if common_chars:
                category_patterns[category] = common_chars

        self.expansion_patterns = dict(category_patterns)

    def _find_common_patterns(self, terms: List[str]) -> List[str]:
        """공통 패턴 찾기"""
        if not terms or len(terms) < 3:
            return []

        patterns = []

        # 공통 접미사 찾기
        suffixes = defaultdict(int)
        for term in terms:
            if len(term) >= 2:
                for i in range(1, min(4, len(term) + 1)):  # 최대 3글자까지
                    suffix = term[-i:]
                    suffixes[suffix] += 1

        # 30% 이상의 용어에서 나타나는 패턴만 선택
        threshold = len(terms) * 0.3
        for suffix, count in suffixes.items():
            if count >= threshold and len(suffix) >= 2:
                patterns.append(suffix)

        return patterns[:5]  # 상위 5개만

    def _build_relationship_graph(self):
        """관계 그래프 구축"""
        print("🕸️ 용어 관계 그래프 구축 중...")

        self.relationship_graph = nx.Graph()

        for term_data in self.terms_data:
            term_name = term_data.get('용어명', '')
            if not term_name:
                continue

            self.relationship_graph.add_node(term_name, **term_data)

            # 동의어 관계
            synonyms = term_data.get('동의어', [])
            for synonym in synonyms:
                if synonym:
                    self.relationship_graph.add_edge(term_name, synonym,
                                                     relation='synonym', weight=1.0)

            # 계층구조 관계
            hierarchy = term_data.get('계층구조', {})
            parents = hierarchy.get('상위개념', [])
            children = hierarchy.get('하위개념', [])

            for parent in parents:
                if parent:
                    self.relationship_graph.add_edge(term_name, parent,
                                                     relation='parent', weight=0.8)

            for child in children:
                if child:
                    self.relationship_graph.add_edge(term_name, child,
                                                     relation='child', weight=0.8)

            # 카테고리 기반 관계
            category = term_data.get('분류', '')
            if category:
                category_terms = [t.get('용어명', '')
                                  for t in self.category_index[category]]
                for related_term in category_terms[:10]:  # 최대 10개까지만
                    if related_term and related_term != term_name:
                        if not self.relationship_graph.has_edge(term_name, related_term):
                            self.relationship_graph.add_edge(term_name, related_term,
                                                             relation='category', weight=0.3)

        print(f"✅ 관계 그래프 구축 완료: {self.relationship_graph.number_of_nodes()}개 노드, "
              f"{self.relationship_graph.number_of_edges()}개 엣지")

    def _analyze_semantic_clusters(self):
        """의미적 클러스터 분석"""
        print("🔬 의미적 클러스터 분석 중...")

        try:
            # 커뮤니티 탐지 알고리즘 사용
            communities = nx.community.greedy_modularity_communities(
                self.relationship_graph)

            cluster_id = 0
            for community in communities:
                if len(community) >= 3:  # 최소 3개 이상의 용어로 구성된 클러스터만
                    cluster_name = f"cluster_{cluster_id}"
                    self.semantic_clusters[cluster_name] = {
                        'terms': list(community),
                        'size': len(community),
                        'dominant_category': self._get_dominant_category(community)
                    }
                    cluster_id += 1

            print(f"✅ {len(self.semantic_clusters)}개 의미적 클러스터 발견")

        except Exception as e:
            print(f"⚠️ 클러스터 분석 실패: {e}")
            self.semantic_clusters = {}

    def _get_dominant_category(self, terms: Set[str]) -> str:
        """클러스터의 지배적 카테고리 찾기"""
        category_counts = Counter()

        for term in terms:
            term_data = self.search_index.get(term)
            if term_data:
                category = term_data.get('분류', '기타')
                category_counts[category] += 1

        if category_counts:
            return category_counts.most_common(1)[0][0]
        return '기타'

    def _build_domain_knowledge_base(self):
        """도메인 지식 베이스 구축 (개선된 매핑)"""
        print("🧠 도메인 지식 베이스 구축 중...")

        self.domain_knowledge_base = {
            # 허증 관련 고급 매핑
            'deficiency_patterns': {
                '血虛': {
                    'primary_herbs': ['當歸', '熟地黃', '白芍', '川芎'],
                    'primary_prescriptions': ['四物湯', '當歸補血湯', '八珍湯'],
                    'related_symptoms': ['面色萎黃', '心悸', '失眠', '月經不調'],
                    'treatment_principles': ['補血', '養血', '調經'],
                    'related_theories': ['營血不足', '心血虛', '肝血虛']
                },
                '氣虛': {
                    'primary_herbs': ['人參', '黃芪', '白朮', '甘草'],
                    'primary_prescriptions': ['四君子湯', '補中益氣湯', '參苓白朮散'],
                    'related_symptoms': ['神疲乏力', '氣短', '自汗', '脫肛'],
                    'treatment_principles': ['補氣', '益氣', '升陽'],
                    'related_theories': ['中氣不足', '脾氣虛', '肺氣虛']
                },
                '陰虛': {
                    'primary_herbs': ['生地黃', '麥門冬', '玄參', '石斛'],
                    'primary_prescriptions': ['六味地黃湯', '麥味地黃湯', '知柏地黃湯'],
                    'related_symptoms': ['潮熱', '盜汗', '五心煩熱', '口燥咽乾'],
                    'treatment_principles': ['滋陰', '養陰', '清熱'],
                    'related_theories': ['腎陰虛', '肺陰虛', '胃陰虛']
                },
                '陽虛': {
                    'primary_herbs': ['附子', '肉桂', '乾薑', '鹿茸'],
                    'primary_prescriptions': ['腎氣丸', '右歸丸', '理中湯'],
                    'related_symptoms': ['畏寒肢冷', '腰膝酸軟', '陽萎', '久瀉'],
                    'treatment_principles': ['溫陽', '助陽', '補陽'],
                    'related_theories': ['腎陽虛', '脾陽虛', '心陽虛']
                }
            },

            # 처방 관계 매핑
            'prescription_relationships': {
                '四物湯': {
                    'base_formula': True,
                    'derived_formulas': ['八物湯', '十全大補湯', '膠艾湯'],
                    'combination_formulas': ['逍遙散', '溫經湯'],
                    'modification_principles': ['加減法', '合方法']
                },
                '四君子湯': {
                    'base_formula': True,
                    'derived_formulas': ['六君子湯', '香砂六君子湯', '參苓白朮散'],
                    'combination_formulas': ['八珍湯', '十全大補湯'],
                    'modification_principles': ['理氣', '化痰', '健脾']
                }
            },

            # 약물 관계 매핑
            'herb_relationships': {
                '人參': {
                    'similar_herbs': ['黨參', '西洋參', '太子參'],
                    'synergistic_herbs': ['黃芪', '白朮', '茯苓'],
                    'antagonistic_herbs': ['萊菔子', '五靈脂'],
                    'processing_methods': ['生曬參', '紅參', '參鬚']
                },
                '當歸': {
                    'similar_herbs': ['川芎', '白芍', '熟地黃'],
                    'synergistic_herbs': ['川芎', '黃芪', '紅花'],
                    'part_usage': ['當歸頭', '當歸身', '當歸尾'],
                    'processing_methods': ['酒當歸', '土炒當歸']
                }
            },

            # 이론 체계 매핑
            'theoretical_frameworks': {
                '陰陽': {
                    'related_concepts': ['陰陽平衡', '陰陽互根', '陰陽轉化'],
                    'clinical_applications': ['寒熱辨證', '虛實辨證', '表裏辨證'],
                    'related_theories': ['五行', '臟腑', '經絡']
                },
                '五行': {
                    'related_concepts': ['五行相生', '五行相克', '五行制化'],
                    'clinical_applications': ['五臟辨證', '情志調攝', '五味調養'],
                    'related_theories': ['臟腑', '經絡', '病機']
                }
            }
        }

    def _save_cache(self):
        """캐시 저장 (개선된 데이터 포함)"""
        try:
            cache_data = {
                'terms_data': self.terms_data,
                'search_index': self.search_index,
                'category_index': dict(self.category_index),
                'synonym_index': self.synonym_index,
                'hierarchical_index': self.hierarchical_index,
                'co_occurrence_matrix': dict(self.co_occurrence_matrix),
                'expansion_patterns': self.expansion_patterns,
                'semantic_clusters': self.semantic_clusters,
                'domain_knowledge_base': self.domain_knowledge_base,
                'relationship_graph_data': nx.node_link_data(self.relationship_graph),
                'timestamp': self._get_current_timestamp()
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")

    def _load_cache(self):
        """캐시 로드 (개선된 데이터 포함)"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.terms_data = cache_data.get('terms_data', [])
            self.search_index = cache_data.get('search_index', {})
            self.category_index = cache_data.get('category_index', {})
            self.synonym_index = cache_data.get('synonym_index', {})
            self.hierarchical_index = cache_data.get('hierarchical_index', {})
            self.co_occurrence_matrix = cache_data.get(
                'co_occurrence_matrix', {})
            self.expansion_patterns = cache_data.get('expansion_patterns', {})
            self.semantic_clusters = cache_data.get('semantic_clusters', {})
            self.domain_knowledge_base = cache_data.get(
                'domain_knowledge_base', {})

            # 관계 그래프 복원
            graph_data = cache_data.get('relationship_graph_data')
            if graph_data:
                self.relationship_graph = nx.node_link_graph(graph_data)
            else:
                self.relationship_graph = nx.Graph()

        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            self._initialize_advanced_basic_index()

    def _initialize_advanced_basic_index(self):
        """고급 기본 인덱스 초기화 (개선된 폴백)"""
        print("📚 고급 기본 중의학 용어로 초기화합니다.")

        # 기본 용어 데이터 (더 풍부한 정보 포함)
        basic_terms = [
            {
                '용어명': '혈허', '용어명_한자': '血虛', '분류': '병증',
                '동의어': ['혈부족', '혈액부족'],
                '검색키워드': ['혈허', '혈부족', '血虛', '혈액부족'],
                '계층구조': {
                    '상위개념': ['허증', '혈병'],
                    '하위개념': ['심혈허', '간혈허']
                }
            },
            {
                '용어명': '기허', '용어명_한자': '氣虛', '분류': '병증',
                '동의어': ['기부족'],
                '검색키워드': ['기허', '기부족', '氣虛'],
                '계층구조': {
                    '상위개념': ['허증', '기병'],
                    '하위개념': ['폐기허', '비기허']
                }
            },
            {
                '용어명': '사물탕', '용어명_한자': '四物湯', '분류': '처방',
                '동의어': [],
                '검색키워드': ['사물탕', '四物湯', '보혈제'],
                '계층구조': {
                    '상위개념': ['보혈제', '방제'],
                    '하위개념': ['가감사물탕']
                }
            },
            {
                '용어명': '인삼', '용어명_한자': '人參', '분류': '약물',
                '동의어': ['고려삼', '홍삼'],
                '검색키워드': ['인삼', '人參', '고려삼', '홍삼', '대보원기'],
                '계층구조': {
                    '상위개념': ['보기약', '본초'],
                    '하위개념': ['생진삼', '홍삼']
                }
            }
        ]

        self.terms_data = basic_terms
        self._build_search_index()
        self._build_category_index()
        self._build_synonym_index()
        self._build_hierarchical_index()

        # 기본 도메인 지식 구축
        self._build_basic_domain_knowledge()

        # 기본 관계 그래프 구축
        self.relationship_graph = nx.Graph()
        for term_data in basic_terms:
            term_name = term_data['용어명']
            self.relationship_graph.add_node(term_name, **term_data)

    def _build_basic_domain_knowledge(self):
        """기본 도메인 지식 구축"""
        self.domain_knowledge_base = {
            'deficiency_patterns': {
                '血虛': {
                    'primary_herbs': ['當歸', '熟地黃', '白芍', '川芎'],
                    'primary_prescriptions': ['四物湯', '當歸補血湯'],
                    'related_symptoms': ['면색위황', '심계', '실면'],
                    'treatment_principles': ['補血', '養血']
                },
                '氣虛': {
                    'primary_herbs': ['人參', '黃芪', '白朮', '甘草'],
                    'primary_prescriptions': ['四君子湯', '補中益氣湯'],
                    'related_symptoms': ['신피핍력', '기단', '자한'],
                    'treatment_principles': ['補氣', '益氣']
                }
            }
        }

    def _get_current_timestamp(self):
        """현재 타임스탬프 반환"""
        return datetime.now().isoformat()

    def expand_query(self, query: str, max_expansions: int = 10) -> List[str]:
        """
        고급 지능형 쿼리 확장 (개선된 버전)
        다층 확장 전략: 직접매칭 → 관계그래프 → 도메인지식 → 의미클러스터 → 공기관계 → 패턴매칭
        """
        expansions = set([query])  # 중복 방지를 위해 set 사용

        try:
            # 1단계: 직접 매칭 및 기본 확장
            if query in self.search_index:
                basic_expansions = self._get_basic_expansions(query)
                expansions.update(basic_expansions)

            # 2단계: 관계 그래프 기반 확장
            if self.relationship_graph.has_node(query):
                graph_expansions = self._get_graph_based_expansions(query)
                expansions.update(graph_expansions)

            # 3단계: 도메인 지식 기반 확장
            domain_expansions = self._get_domain_knowledge_expansions(query)
            expansions.update(domain_expansions)

            # 4단계: 의미적 클러스터 기반 확장
            cluster_expansions = self._get_cluster_based_expansions(query)
            expansions.update(cluster_expansions)

            # 5단계: 공기 관계 기반 확장
            co_occurrence_expansions = self._get_co_occurrence_expansions(
                query)
            expansions.update(co_occurrence_expansions)

            # 6단계: 패턴 기반 확장
            pattern_expansions = self._get_pattern_based_expansions(query)
            expansions.update(pattern_expansions)

            # 7단계: 지능적 분할 및 조합
            split_expansions = self.split_query_intelligently(query)
            expansions.update(split_expansions)

            # 품질 필터링 및 순위 매기기
            filtered_expansions = self._filter_and_rank_expansions(
                query, list(expansions))

            return filtered_expansions[:max_expansions]

        except Exception as e:
            print(f"⚠️ 고급 쿼리 확장 실패: {e}")
            # 폴백: 기본 확장
            return self._get_basic_expansions(query)[:max_expansions]

    def _get_basic_expansions(self, query: str) -> List[str]:
        """기본 확장 (1단계)"""
        expansions = []

        if query in self.search_index:
            term_data = self.search_index[query]

            # 동의어 추가
            synonyms = term_data.get('동의어', [])
            expansions.extend(synonyms[:3])

            # 한자/한글명 상호 추가
            hanja = term_data.get('용어명_한자', '')
            hangul = term_data.get('용어명', '')

            if hanja and hanja != query:
                expansions.append(hanja)
            if hangul and hangul != query:
                expansions.append(hangul)

            # 검색키워드 추가
            keywords = term_data.get('검색키워드', [])
            expansions.extend([k for k in keywords[:3] if k != query])

        return expansions

    def _get_graph_based_expansions(self, query: str) -> List[str]:
        """관계 그래프 기반 확장 (2단계)"""
        expansions = []
        try:
            # 직접 이웃 노드들
            neighbors = list(self.relationship_graph.neighbors(query))

            # 관계별 가중치 적용
            weighted_neighbors = []
            for neighbor in neighbors:
                edge_data = self.relationship_graph.get_edge_data(
                    query, neighbor)
                relation = edge_data.get('relation', 'unknown')
                weight = edge_data.get('weight', 0.5)

                # 관계별 우선순위
                priority_map = {
                    'synonym': 1.0,
                    'parent': 0.8,
                    'child': 0.8,
                    'category': 0.3
                }

                final_weight = weight * priority_map.get(relation, 0.1)
                weighted_neighbors.append((neighbor, final_weight))

            # 가중치순 정렬 후 상위 항목 선택
            weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
            expansions.extend(
                [neighbor for neighbor, _ in weighted_neighbors[:5]])

            # 2홉 이웃도 고려 (가중치 감소)
            if len(expansions) < 3:
                two_hop_neighbors = []
                for neighbor in neighbors[:3]:
                    second_neighbors = list(
                        self.relationship_graph.neighbors(neighbor))
                    two_hop_neighbors.extend(
                        [n for n in second_neighbors if n != query])

                expansions.extend(two_hop_neighbors[:2])

        except Exception as e:
            print(f"⚠️ 그래프 기반 확장 실패: {e}")

        return expansions

    def _get_domain_knowledge_expansions(self, query: str) -> List[str]:
        """도메인 지식 기반 확장 (3단계)"""
        expansions = []

        try:
            # 허증 패턴 매칭
            deficiency_patterns = self.domain_knowledge_base.get(
                'deficiency_patterns', {})
            if query in deficiency_patterns:
                pattern_data = deficiency_patterns[query]
                expansions.extend(pattern_data.get('primary_herbs', [])[:2])
                expansions.extend(pattern_data.get(
                    'primary_prescriptions', [])[:2])
                expansions.extend(pattern_data.get('related_theories', [])[:1])

            # 처방 관계 매핑
            prescription_relationships = self.domain_knowledge_base.get(
                'prescription_relationships', {})
            if query in prescription_relationships:
                relationship_data = prescription_relationships[query]
                expansions.extend(relationship_data.get(
                    'derived_formulas', [])[:2])
                expansions.extend(relationship_data.get(
                    'combination_formulas', [])[:1])

            # 약물 관계 매핑
            herb_relationships = self.domain_knowledge_base.get(
                'herb_relationships', {})
            if query in herb_relationships:
                herb_data = herb_relationships[query]
                expansions.extend(herb_data.get('similar_herbs', [])[:2])
                expansions.extend(herb_data.get('synergistic_herbs', [])[:1])

            # 이론 체계 매핑
            theoretical_frameworks = self.domain_knowledge_base.get(
                'theoretical_frameworks', {})
            if query in theoretical_frameworks:
                theory_data = theoretical_frameworks[query]
                expansions.extend(theory_data.get('related_concepts', [])[:2])

        except Exception as e:
            print(f"⚠️ 도메인 지식 확장 실패: {e}")

        return expansions

    def _get_cluster_based_expansions(self, query: str) -> List[str]:
        """의미적 클러스터 기반 확장 (4단계)"""
        expansions = []

        try:
            # 쿼리가 속한 클러스터 찾기
            query_clusters = []
            for cluster_name, cluster_data in self.semantic_clusters.items():
                if query in cluster_data['terms']:
                    query_clusters.append(cluster_data)

            # 같은 클러스터 내의 다른 용어들 추가
            for cluster_data in query_clusters:
                cluster_terms = cluster_data['terms']
                related_terms = [
                    term for term in cluster_terms if term != query]
                expansions.extend(related_terms[:3])  # 최대 3개까지

        except Exception as e:
            print(f"⚠️ 클러스터 기반 확장 실패: {e}")

        return expansions

    def _get_co_occurrence_expansions(self, query: str) -> List[str]:
        """공기 관계 기반 확장 (5단계)"""
        expansions = []

        try:
            if query in self.co_occurrence_matrix:
                co_occurrences = self.co_occurrence_matrix[query]
                # 빈도 순으로 정렬
                sorted_co_occurrences = sorted(co_occurrences.items(),
                                               key=lambda x: x[1], reverse=True)
                expansions.extend(
                    [term for term, _ in sorted_co_occurrences[:3]])

        except Exception as e:
            print(f"⚠️ 공기 관계 확장 실패: {e}")

        return expansions

    def _get_pattern_based_expansions(self, query: str) -> List[str]:
        """패턴 기반 확장 (6단계)"""
        expansions = []

        try:
            # 쿼리의 카테고리 확인
            term_data = self.search_index.get(query)
            if term_data:
                category = term_data.get('분류', '')
                if category in self.expansion_patterns:
                    patterns = self.expansion_patterns[category]

                    # 같은 패턴을 가진 다른 용어들 찾기
                    for pattern in patterns:
                        if pattern in query:
                            # 같은 패턴의 다른 용어들 검색
                            pattern_terms = [term for term in self.search_index.keys()
                                             if pattern in term and term != query]
                            expansions.extend(pattern_terms[:2])

        except Exception as e:
            print(f"⚠️ 패턴 기반 확장 실패: {e}")

        return expansions

    def _filter_and_rank_expansions(self, query: str, expansions: List[str]) -> List[str]:
        """확장 결과 필터링 및 순위 매기기"""
        if not expansions:
            return []

        scored_expansions = []

        for expansion in expansions:
            if expansion == query:
                continue

            score = 0.0

            # 1. 길이 유사성 (너무 다르면 관련성 낮음)
            length_diff = abs(len(expansion) - len(query))
            if length_diff <= 2:
                score += 1.0
            elif length_diff <= 4:
                score += 0.5

            # 2. 글자 겹침 정도
            common_chars = set(query) & set(expansion)
            overlap_ratio = len(common_chars) / \
                max(len(set(query)), len(set(expansion)))
            score += overlap_ratio * 2.0

            # 3. 카테고리 유사성
            query_data = self.search_index.get(query)
            expansion_data = self.search_index.get(expansion)

            if query_data and expansion_data:
                query_category = query_data.get('분류', '')
                expansion_category = expansion_data.get('분류', '')

                if query_category == expansion_category:
                    score += 1.5
                elif query_category and expansion_category:
                    # 관련 카테고리인지 확인
                    related_categories = {
                        '병증': ['증상', '징후'],
                        '처방': ['치법'],
                        '약물': ['본초'],
                        '생리': ['병리', '변증']
                    }

                    if expansion_category in related_categories.get(query_category, []):
                        score += 0.8

            # 4. 관계 그래프에서의 거리
            if (self.relationship_graph.has_node(query) and
                    self.relationship_graph.has_node(expansion)):
                try:
                    distance = nx.shortest_path_length(
                        self.relationship_graph, query, expansion)
                    if distance == 1:
                        score += 2.0
                    elif distance == 2:
                        score += 1.0
                    elif distance <= 3:
                        score += 0.5
                except nx.NetworkXNoPath:
                    pass

            # 5. 의미있는 용어인지 확인
            if self._is_meaningful_expansion(expansion):
                score += 0.5

            scored_expansions.append((expansion, score))

        # 점수순 정렬
        scored_expansions.sort(key=lambda x: x[1], reverse=True)

        # 중복 제거하면서 결과 반환
        seen = set()
        filtered_result = []

        for expansion, score in scored_expansions:
            if expansion not in seen and score > 0.5:  # 최소 점수 임계값
                seen.add(expansion)
                filtered_result.append(expansion)

        return filtered_result

    def _is_meaningful_expansion(self, term: str) -> bool:
        """의미있는 확장인지 판단"""
        if len(term) < 2:
            return False

        # 한자 비율 확인
        chinese_char_count = sum(
            1 for char in term if '\u4e00' <= char <= '\u9fff')
        if chinese_char_count / len(term) < 0.5:
            return False

        # 표준용어집에 있는지 확인
        return term in self.search_index

    def split_query_intelligently(self, query: str) -> List[str]:
        """고급 지능적 쿼리 분할"""
        parts = [query]

        try:
            # 길이별 분할 전략 (개선됨)
            if len(query) >= 4:
                # 의미 단위 분할 시도
                meaningful_parts = self._extract_meaningful_subterms(query)
                parts.extend(meaningful_parts)

                # 길이별 분할
                for i in range(len(query) - 1):
                    if i + 2 <= len(query):
                        part = query[i:i + 2]
                        if self._is_meaningful_expansion(part):
                            parts.append(part)

                for i in range(len(query) - 2):
                    if i + 3 <= len(query):
                        part = query[i:i + 3]
                        if self._is_meaningful_expansion(part):
                            parts.append(part)

            elif len(query) == 3:
                # 3글자: 앞 2글자, 뒤 2글자 추가
                front_part = query[:2]
                back_part = query[1:]

                if self._is_meaningful_expansion(front_part):
                    parts.append(front_part)
                if self._is_meaningful_expansion(back_part):
                    parts.append(back_part)

            # 중복 제거 및 의미있는 부분만 선택
            unique_parts = []
            for part in parts:
                if part not in unique_parts and self._is_meaningful_expansion(part):
                    unique_parts.append(part)

            return unique_parts

        except Exception as e:
            print(f"⚠️ 지능적 분할 실패: {e}")
            return [query]

    def _extract_meaningful_subterms(self, query: str) -> List[str]:
        """의미 있는 하위 용어 추출"""
        subterms = []

        # 알려진 패턴 기반 분할
        common_suffixes = ['湯', '散', '丸', '膏',
                           '證', '病', '症', '虛', '實', '熱', '寒']

        for suffix in common_suffixes:
            if query.endswith(suffix) and len(query) > len(suffix):
                base_part = query[:-len(suffix)]
                if len(base_part) >= 2 and base_part in self.search_index:
                    subterms.append(base_part)
                if suffix in self.search_index:
                    subterms.append(suffix)

        # 복합어 분할 시도
        if len(query) >= 4:
            for i in range(2, len(query)):
                left_part = query[:i]
                right_part = query[i:]

                if (len(left_part) >= 2 and len(right_part) >= 2 and
                        left_part in self.search_index and right_part in self.search_index):
                    subterms.extend([left_part, right_part])

        return subterms

    def get_related_terms(self, query: str, max_terms: int = 10) -> List[str]:
        """고급 관련 용어 반환"""
        related = []

        try:
            # 다양한 방법으로 관련 용어 수집
            methods = [
                self._get_basic_related_terms,
                self._get_graph_related_terms,
                self._get_domain_related_terms,
                self._get_cluster_related_terms
            ]

            for method in methods:
                method_results = method(query)
                related.extend(method_results)

            # 중복 제거 및 순위 매기기
            unique_related = []
            seen = set()

            for term in related:
                if term and term != query and term not in seen:
                    seen.add(term)
                    unique_related.append(term)

            return unique_related[:max_terms]

        except Exception as e:
            print(f"⚠️ 관련 용어 추출 실패: {e}")
            return []

    def _get_basic_related_terms(self, query: str) -> List[str]:
        """기본 관련 용어"""
        related = []

        if query in self.search_index:
            term_data = self.search_index[query]

            # 동의어
            synonyms = term_data.get('동의어', [])
            related.extend(synonyms[:3])

            # 같은 분류의 다른 용어들
            category = term_data.get('분류', '')
            if category in self.category_index:
                category_terms = self.category_index[category]
                for cat_term in category_terms[:5]:
                    term_name = cat_term.get('용어명', '')
                    if term_name and term_name != query:
                        related.append(term_name)

        return related

    def _get_graph_related_terms(self, query: str) -> List[str]:
        """그래프 기반 관련 용어"""
        related = []

        if self.relationship_graph.has_node(query):
            neighbors = list(self.relationship_graph.neighbors(query))
            related.extend(neighbors[:5])

        return related

    def _get_domain_related_terms(self, query: str) -> List[str]:
        """도메인 지식 기반 관련 용어"""
        related = []

        # 도메인 지식에서 관련 용어 추출
        for pattern_type, patterns in self.domain_knowledge_base.items():
            if isinstance(patterns, dict) and query in patterns:
                pattern_data = patterns[query]
                if isinstance(pattern_data, dict):
                    for key, values in pattern_data.items():
                        if isinstance(values, list):
                            related.extend(values[:2])

        return related

    def _get_cluster_related_terms(self, query: str) -> List[str]:
        """클러스터 기반 관련 용어"""
        related = []

        for cluster_data in self.semantic_clusters.values():
            if query in cluster_data['terms']:
                cluster_terms = [
                    term for term in cluster_data['terms'] if term != query]
                related.extend(cluster_terms[:3])

        return related

    def get_term_info(self, term: str) -> Optional[Dict]:
        """특정 용어의 상세 정보 반환"""
        try:
            if term in self.search_index:
                base_info = self.search_index[term].copy()

                # 추가 정보 enrichment
                enriched_info = base_info.copy()

                # 그래프 정보 추가
                if self.relationship_graph.has_node(term):
                    neighbors = list(self.relationship_graph.neighbors(term))
                    enriched_info['graph_neighbors'] = neighbors[:10]
                    enriched_info['graph_degree'] = self.relationship_graph.degree(
                        term)

                # 클러스터 정보 추가
                for cluster_name, cluster_data in self.semantic_clusters.items():
                    if term in cluster_data['terms']:
                        enriched_info['semantic_cluster'] = cluster_name
                        enriched_info['cluster_size'] = cluster_data['size']
                        break

                # 도메인 지식 정보 추가
                domain_info = {}
                for pattern_type, patterns in self.domain_knowledge_base.items():
                    if isinstance(patterns, dict) and term in patterns:
                        domain_info[pattern_type] = patterns[term]

                if domain_info:
                    enriched_info['domain_knowledge'] = domain_info

                return enriched_info

            return None
        except Exception as e:
            print(f"⚠️ 용어 정보 조회 실패: {e}")
            return None

    def search_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """분류별 용어 검색 (개선된 버전)"""
        try:
            if category in self.category_index:
                terms = self.category_index[category]

                # 용어별 점수 계산 (그래프 중심성, 클러스터 크기 등 고려)
                scored_terms = []
                for term_data in terms:
                    term_name = term_data.get('용어명', '')
                    score = 1.0  # 기본 점수

                    # 그래프 중심성 고려
                    if self.relationship_graph.has_node(term_name):
                        degree = self.relationship_graph.degree(term_name)
                        score += degree * 0.1

                    # 클러스터 크기 고려
                    for cluster_data in self.semantic_clusters.values():
                        if term_name in cluster_data['terms']:
                            score += cluster_data['size'] * 0.05
                            break

                    scored_terms.append((term_data, score))

                # 점수순 정렬
                scored_terms.sort(key=lambda x: x[1], reverse=True)

                return [term_data for term_data, _ in scored_terms[:limit]]

            return []
        except Exception as e:
            print(f"⚠️ 카테고리 검색 실패: {e}")
            return []

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """고급 유사 용어 검색"""
        matches = []

        try:
            for term in self.search_index.keys():
                if term == query:
                    continue

                similarity = self._calculate_term_similarity(query, term)

                if similarity >= threshold:
                    matches.append((term, similarity))

            # 유사도 순으로 정렬
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:20]  # 상위 20개

        except Exception as e:
            print(f"⚠️ 유사 검색 실패: {e}")
            return []

    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        """용어 간 유사도 계산 (다차원)"""
        similarity = 0.0

        # 1. 문자 겹침 유사도
        common_chars = set(term1) & set(term2)
        char_similarity = len(common_chars) / max(len(term1), len(term2))
        similarity += char_similarity * 0.4

        # 2. 길이 유사도
        length_diff = abs(len(term1) - len(term2))
        max_length = max(len(term1), len(term2))
        length_similarity = 1.0 - (length_diff / max_length)
        similarity += length_similarity * 0.2

        # 3. 카테고리 유사도
        term1_data = self.search_index.get(term1)
        term2_data = self.search_index.get(term2)

        if term1_data and term2_data:
            cat1 = term1_data.get('분류', '')
            cat2 = term2_data.get('분류', '')

            if cat1 == cat2:
                similarity += 0.3
            elif cat1 and cat2:
                # 관련 카테고리 확인
                related_categories = {
                    '병증': ['증상', '징후'],
                    '처방': ['치법'],
                    '약물': ['본초']
                }

                if cat2 in related_categories.get(cat1, []):
                    similarity += 0.15

        # 4. 그래프 거리 유사도
        if (self.relationship_graph.has_node(term1) and
                self.relationship_graph.has_node(term2)):
            try:
                distance = nx.shortest_path_length(
                    self.relationship_graph, term1, term2)
                if distance <= 3:
                    graph_similarity = 1.0 / (distance + 1)
                    similarity += graph_similarity * 0.1
            except nx.NetworkXNoPath:
                pass

        return min(similarity, 1.0)

    def get_statistics(self) -> Dict:
        """고급 용어집 통계 정보 반환"""
        try:
            stats = {
                'total_terms': len(self.terms_data),
                'search_index_size': len(self.search_index),
                'categories': len(self.category_index),
                'terms_with_synonyms': len(self.synonym_index),
                'terms_with_hierarchy': len(self.hierarchical_index),
                'relationship_graph_nodes': self.relationship_graph.number_of_nodes(),
                'relationship_graph_edges': self.relationship_graph.number_of_edges(),
                'semantic_clusters': len(self.semantic_clusters),
                'domain_knowledge_patterns': len(self.domain_knowledge_base),
                'expansion_patterns': len(self.expansion_patterns)
            }

            # 카테고리별 분포
            category_distribution = {}
            for category, terms in self.category_index.items():
                category_distribution[category] = len(terms)

            stats['category_distribution'] = category_distribution

            # 그래프 통계
            if self.relationship_graph.number_of_nodes() > 0:
                stats['graph_density'] = nx.density(self.relationship_graph)
                stats['average_clustering'] = nx.average_clustering(
                    self.relationship_graph)

                # 중심성 높은 용어들
                centrality = nx.degree_centrality(self.relationship_graph)
                top_central_terms = sorted(
                    centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                stats['most_central_terms'] = [
                    term for term, _ in top_central_terms]

            return stats

        except Exception as e:
            print(f"⚠️ 통계 정보 생성 실패: {e}")
            return {'error': str(e)}

    def analyze_query_patterns(self, queries: List[str]) -> Dict:
        """쿼리 패턴 분석"""
        analysis = {
            'total_queries': len(queries),
            'unique_queries': len(set(queries)),
            'category_distribution': Counter(),
            'length_distribution': Counter(),
            'expansion_effectiveness': {},
            'common_patterns': []
        }

        for query in queries:
            # 길이 분포
            analysis['length_distribution'][len(query)] += 1

            # 카테고리 분포
            if query in self.search_index:
                category = self.search_index[query].get('분류', '기타')
                analysis['category_distribution'][category] += 1

            # 확장 효과 분석
            expansions = self.expand_query(query, max_expansions=5)
            analysis['expansion_effectiveness'][query] = len(expansions)

        # 공통 패턴 추출
        pattern_counter = Counter()
        for query in queries:
            if len(query) >= 2:
                for i in range(len(query) - 1):
                    pattern = query[i:i + 2]
                    pattern_counter[pattern] += 1

        analysis['common_patterns'] = pattern_counter.most_common(10)

        return analysis

    def export_knowledge_graph(self, format: str = 'gexf') -> str:
        """지식 그래프 내보내기"""
        try:
            output_file = self.cache_path / f'knowledge_graph.{format}'

            if format == 'gexf':
                nx.write_gexf(self.relationship_graph, output_file)
            elif format == 'gml':
                nx.write_gml(self.relationship_graph, output_file)
            elif format == 'graphml':
                nx.write_graphml(self.relationship_graph, output_file)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")

            return str(output_file)

        except Exception as e:
            print(f"⚠️ 지식 그래프 내보내기 실패: {e}")
            return ""

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
            print("🔄 표준용어집 고급 인덱스 강제 재구축 중...")
            self._build_index()
            self._build_relationship_graph()
            self._analyze_semantic_clusters()
            self._build_domain_knowledge_base()
            self._save_cache()
            print("✅ 고급 인덱스 재구축 완료")
        except Exception as e:
            print(f"⚠️ 인덱스 재구축 실패: {e}")

    def validate_system_integrity(self) -> Dict:
        """시스템 무결성 검증"""
        validation = {
            'basic_indexes_ok': True,
            'graph_ok': True,
            'clusters_ok': True,
            'domain_knowledge_ok': True,
            'errors': [],
            'warnings': []
        }

        try:
            # 기본 인덱스 검증
            if not self.search_index:
                validation['basic_indexes_ok'] = False
                validation['errors'].append("검색 인덱스가 비어있습니다")

            # 그래프 검증
            if self.relationship_graph.number_of_nodes() == 0:
                validation['graph_ok'] = False
                validation['warnings'].append("관계 그래프가 비어있습니다")

            # 클러스터 검증
            if not self.semantic_clusters:
                validation['clusters_ok'] = False
                validation['warnings'].append("의미적 클러스터가 없습니다")

            # 도메인 지식 검증
            if not self.domain_knowledge_base:
                validation['domain_knowledge_ok'] = False
                validation['warnings'].append("도메인 지식이 비어있습니다")

            # 데이터 일관성 검증
            inconsistencies = self._check_data_consistency()
            if inconsistencies:
                validation['warnings'].extend(inconsistencies)

        except Exception as e:
            validation['errors'].append(f"검증 중 오류: {e}")

        return validation

    def _check_data_consistency(self) -> List[str]:
        """데이터 일관성 검사"""
        issues = []

        # 검색 인덱스와 원본 데이터 일치 확인
        expected_terms = set()
        for term_data in self.terms_data:
            term_name = term_data.get('용어명', '')
            if term_name:
                expected_terms.add(term_name)

        actual_terms = set(term for term in self.search_index.keys())

        if len(expected_terms) > len(actual_terms) * 0.8:  # 80% 이상 일치해야 함
            missing = expected_terms - actual_terms
            if missing:
                issues.append(f"검색 인덱스에서 누락된 용어: {len(missing)}개")

        return issues


# 편의 함수들
def create_terms_manager() -> MedicalTermsManager:
    """표준용어집 관리자 생성 편의 함수"""
    return MedicalTermsManager()


def test_advanced_terms_manager():
    """고급 테스트용 함수"""
    print("🧪 고급 표준용어집 관리자 테스트")

    manager = MedicalTermsManager()

    # 기본 통계 정보 출력
    stats = manager.get_statistics()
    print(f"\n📊 고급 시스템 통계:")
    print(f"   총 용어 수: {stats.get('total_terms', 0):,}개")
    print(f"   검색 인덱스: {stats.get('search_index_size', 0):,}개")
    print(f"   카테고리: {stats.get('categories', 0)}개")
    print(f"   관계 그래프 노드: {stats.get('relationship_graph_nodes', 0):,}개")
    print(f"   관계 그래프 엣지: {stats.get('relationship_graph_edges', 0):,}개")
    print(f"   의미적 클러스터: {stats.get('semantic_clusters', 0)}개")
    print(f"   도메인 지식 패턴: {stats.get('domain_knowledge_patterns', 0)}개")

    # 그래프 통계
    if 'graph_density' in stats:
        print(f"   그래프 밀도: {stats['graph_density']:.4f}")
        print(f"   평균 클러스터링: {stats['average_clustering']:.4f}")
        print(
            f"   중심성 높은 용어: {', '.join(stats.get('most_central_terms', []))}")

    # 고급 쿼리 확장 테스트
    test_queries = ['血虛', '四君子湯', '人參', '陰虛', '補中益氣湯']

    print(f"\n🔍 고급 쿼리 확장 테스트:")
    for query in test_queries:
        print(f"\n   📝 '{query}' 확장 테스트:")

        # 기본 확장
        expansions = manager.expand_query(query, max_expansions=8)
        print(f"      확장 결과 ({len(expansions)}개): {', '.join(expansions)}")

        # 관련 용어
        related = manager.get_related_terms(query, max_terms=5)
        if related:
            print(f"      관련 용어: {', '.join(related)}")

        # 용어 정보
        term_info = manager.get_term_info(query)
        if term_info:
            category = term_info.get('분류', '미분류')
            print(f"      분류: {category}")

            if 'graph_degree' in term_info:
                print(f"      그래프 연결도: {term_info['graph_degree']}")

            if 'semantic_cluster' in term_info:
                print(f"      소속 클러스터: {term_info['semantic_cluster']}")

    # 유사 검색 테스트
    print(f"\n🔎 유사 검색 테스트 ('血虛'와 유사한 용어):")
    similar_terms = manager.fuzzy_search('血虛', threshold=0.3)
    for term, similarity in similar_terms[:5]:
        print(f"      {term}: {similarity:.3f}")

    # 시스템 무결성 검증
    print(f"\n🔧 시스템 무결성 검증:")
    validation = manager.validate_system_integrity()

    status_symbols = {True: "✅", False: "❌"}
    print(f"   기본 인덱스: {status_symbols[validation['basic_indexes_ok']]}")
    print(f"   관계 그래프: {status_symbols[validation['graph_ok']]}")
    print(f"   의미 클러스터: {status_symbols[validation['clusters_ok']]}")
    print(f"   도메인 지식: {status_symbols[validation['domain_knowledge_ok']]}")

    if validation['errors']:
        print(f"   ⚠️ 오류: {', '.join(validation['errors'])}")
    if validation['warnings']:
        print(f"   💡 경고: {', '.join(validation['warnings'])}")

    # 도메인 지식 테스트
    print(f"\n🧠 도메인 지식 베이스 테스트:")
    if manager.domain_knowledge_base:
        deficiency_patterns = manager.domain_knowledge_base.get(
            'deficiency_patterns', {})
        if deficiency_patterns:
            print(f"   허증 패턴: {len(deficiency_patterns)}개")
            for pattern_name in list(deficiency_patterns.keys())[:3]:
                pattern_data = deficiency_patterns[pattern_name]
                herbs = pattern_data.get('primary_herbs', [])
                prescriptions = pattern_data.get('primary_prescriptions', [])
                print(
                    f"      {pattern_name}: 주요약재 {len(herbs)}개, 주요처방 {len(prescriptions)}개")


def benchmark_expansion_performance():
    """확장 성능 벤치마크"""
    import time

    print("⚡ 확장 성능 벤치마크 테스트")

    manager = MedicalTermsManager()

    test_queries = [
        '血虛', '氣虛', '陰虛', '陽虛', '四物湯', '四君子湯', '六君子湯',
        '補中益氣湯', '當歸補血湯', '人參', '當歸', '黃芪', '白朮', '茯苓',
        '心悸', '失眠', '眩暈', '頭痛', '胸痛', '腹痛'
    ]

    # 단일 확장 성능 테스트
    print("\n📊 단일 쿼리 확장 성능:")
    total_time = 0
    total_expansions = 0

    for query in test_queries[:10]:  # 상위 10개만 테스트
        start_time = time.time()
        expansions = manager.expand_query(query, max_expansions=10)
        end_time = time.time()

        query_time = end_time - start_time
        total_time += query_time
        total_expansions += len(expansions)

        print(f"   {query}: {len(expansions)}개 확장, {query_time:.4f}초")

    avg_time = total_time / len(test_queries[:10])
    avg_expansions = total_expansions / len(test_queries[:10])

    print(f"\n📈 성능 요약:")
    print(f"   평균 처리 시간: {avg_time:.4f}초")
    print(f"   평균 확장 수: {avg_expansions:.1f}개")
    print(f"   초당 처리 가능 쿼리: {1 / avg_time:.1f}개")

    # 배치 처리 성능 테스트
    print(f"\n🔄 배치 처리 성능 (20개 쿼리):")
    start_time = time.time()

    batch_results = []
    for query in test_queries:
        expansions = manager.expand_query(query, max_expansions=5)
        batch_results.append((query, expansions))

    batch_time = time.time() - start_time

    print(f"   총 처리 시간: {batch_time:.4f}초")
    print(f"   평균 쿼리당 시간: {batch_time / len(test_queries):.4f}초")
    print(f"   처리량: {len(test_queries) / batch_time:.1f} 쿼리/초")


def export_analysis_report(manager: MedicalTermsManager, output_path: str = None):
    """분석 리포트 내보내기"""
    if output_path is None:
        output_path = manager.cache_path / 'analysis_report.txt'

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("표준한의학용어집 고급 분석 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 기본 통계
            stats = manager.get_statistics()
            f.write("📊 기본 통계\n")
            f.write("-" * 40 + "\n")
            for key, value in stats.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            # 도메인 지식 분석
            f.write("🧠 도메인 지식 베이스 분석\n")
            f.write("-" * 40 + "\n")

            deficiency_patterns = manager.domain_knowledge_base.get(
                'deficiency_patterns', {})
            f.write(f"허증 패턴: {len(deficiency_patterns)}개\n")

            for pattern_name, pattern_data in deficiency_patterns.items():
                f.write(f"\n{pattern_name}:\n")
                f.write(
                    f"  주요 약재: {', '.join(pattern_data.get('primary_herbs', []))}\n")
                f.write(
                    f"  주요 처방: {', '.join(pattern_data.get('primary_prescriptions', []))}\n")
                f.write(
                    f"  관련 증상: {', '.join(pattern_data.get('related_symptoms', []))}\n")

            # 의미적 클러스터 분석
            f.write(f"\n🔬 의미적 클러스터 분석\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 클러스터 수: {len(manager.semantic_clusters)}\n")

            for cluster_name, cluster_data in manager.semantic_clusters.items():
                f.write(f"\n{cluster_name}:\n")
                f.write(f"  크기: {cluster_data['size']}개 용어\n")
                f.write(f"  지배적 카테고리: {cluster_data['dominant_category']}\n")
                f.write(f"  용어들: {', '.join(cluster_data['terms'][:10])}\n")
                if len(cluster_data['terms']) > 10:
                    f.write(f"  ... 외 {len(cluster_data['terms']) - 10}개\n")

            # 시스템 무결성 검증
            f.write(f"\n🔧 시스템 무결성 검증\n")
            f.write("-" * 40 + "\n")

            validation = manager.validate_system_integrity()
            f.write(
                f"기본 인덱스: {'정상' if validation['basic_indexes_ok'] else '오류'}\n")
            f.write(f"관계 그래프: {'정상' if validation['graph_ok'] else '오류'}\n")
            f.write(
                f"의미 클러스터: {'정상' if validation['clusters_ok'] else '오류'}\n")
            f.write(
                f"도메인 지식: {'정상' if validation['domain_knowledge_ok'] else '오류'}\n")

            if validation['errors']:
                f.write(f"\n오류 목록:\n")
                for error in validation['errors']:
                    f.write(f"  - {error}\n")

            if validation['warnings']:
                f.write(f"\n경고 목록:\n")
                for warning in validation['warnings']:
                    f.write(f"  - {warning}\n")

        print(f"📄 분석 리포트가 저장되었습니다: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"⚠️ 리포트 생성 실패: {e}")
        return None


if __name__ == "__main__":
    print("🚀 고급 표준한의학용어집 관리자 테스트 시작")

    # 기본 테스트
    test_advanced_terms_manager()

    print("\n" + "=" * 60)

    # 성능 벤치마크
    benchmark_expansion_performance()

    print("\n" + "=" * 60)

    # 분석 리포트 생성
    manager = create_terms_manager()
    report_path = export_analysis_report(manager)

    if report_path:
        print(f"✅ 모든 테스트 완료! 상세 리포트: {report_path}")
    else:
        print("⚠️ 일부 테스트에서 문제가 발생했습니다.")
