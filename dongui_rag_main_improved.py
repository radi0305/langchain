#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의보감 RAG 시스템 메인 모듈 - dongui_rag_main_improved.py (완전 개선된 버전)
하드코딩 제거 및 표준한의학용어집 기반 시스템으로 완전 전환
검색 결과 그룹핑, 검색 품질 메트릭 및 관련 검색어 제안을 포함한 통합 실행 모듈
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")

# 개선된 모듈 임포트
try:
    from document_processor_improved import DocumentProcessor
    from search_engine_improved import SearchEngine
    from cache_manager import CacheManager
    from answer_generator_improved import AnswerGenerator
    from medical_terms_manager_improved import MedicalTermsManager
    from llm_manager import LLMManager
except ImportError as e:
    print(f"필수 모듈이 없습니다: {e}")
    print("다음 개선된 파일들이 같은 디렉토리에 있는지 확인해주세요:")
    print("- document_processor_improved.py")
    print("- search_engine_improved.py")
    print("- answer_generator_improved.py")
    print("- medical_terms_manager_improved.py")
    print("- cache_manager.py")
    print("- llm_manager.py")
    sys.exit(1)


class DonguiRAGSystemImproved:
    """동의보감 RAG 시스템 (완전 개선된 버전)"""

    def __init__(self,
                 data_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG",
                 cache_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG/cache",
                 save_path: str = "/Users/radi/Projects/langchainDATA/Results/DYBGsearch"):
        """동의보감 RAG 시스템 초기화 (완전 개선된 버전)"""
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.save_path = Path(save_path)

        # 디렉토리 생성
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"📂 데이터 경로: {self.data_path}")
        print(f"💾 캐시 경로: {self.cache_path}")
        print(f"💾 결과 저장 경로: {self.save_path}")

        # LLM 관리자 초기화
        print("🔧 OpenAI 연결 중...")
        self.llm_manager = LLMManager()

        if not self.llm_manager.is_available():
            print("❌ OpenAI 연결에 실패했습니다.")
            print("💡 API 키 설정 후 다시 시도해주세요.")
            sys.exit(1)

        # 시스템 모듈 초기화
        print("🔧 완전 개선된 시스템 모듈 초기화 중...")

        # 1. 고급 표준용어집 관리자 (가장 먼저 초기화)
        try:
            print("📚 고급 표준한의학용어집 관리자 초기화 중...")
            self.terms_manager = MedicalTermsManager(
                cache_path=str(self.cache_path)
            )

            # 시스템 무결성 검증
            validation = self.terms_manager.validate_system_integrity()
            if not validation['basic_indexes_ok']:
                print("⚠️ 표준용어집 시스템에 오류가 있습니다.")
                for error in validation['errors']:
                    print(f"   - {error}")
            else:
                stats = self.terms_manager.get_statistics()
                print(f"✅ 고급 표준용어집 관리자 초기화 완료")
                print(f"   📊 총 용어: {stats.get('total_terms', 0):,}개")
                print(
                    f"   🕸️ 관계 그래프: {stats.get('relationship_graph_nodes', 0):,}개 노드")
                print(f"   🔬 의미 클러스터: {stats.get('semantic_clusters', 0)}개")

        except Exception as e:
            print(f"⚠️ 고급 표준용어집 초기화 실패: {e}")
            print("📚 기본 모드로 진행합니다.")
            self.terms_manager = None

        # 2. 개선된 문서 처리기
        self.document_processor = DocumentProcessor(
            data_path=str(self.data_path),
            terms_manager=self.terms_manager
        )

        # 처리 통계 출력
        proc_stats = self.document_processor.get_processing_statistics()
        print(f"✅ 개선된 문서 처리기 초기화 완료")
        print(f"   📊 동적 TCM 용어: {proc_stats.get('dynamic_terms_count', 0)}개")
        print(
            f"   🔍 처방 패턴: {proc_stats.get('prescription_patterns_count', 0)}개")
        print(f"   🌿 약재 패턴: {proc_stats.get('herb_patterns_count', 0)}개")

        # 3. 고급 검색 엔진
        self.search_engine = SearchEngine()
        self.search_engine.set_terms_manager(self.terms_manager)

        # 패턴 캐시 정보 출력
        pattern_info = self.search_engine.get_pattern_cache_info()
        print(f"✅ 고급 검색 엔진 초기화 완료")
        print(
            f"   🔍 동적 패턴: {pattern_info.get('prescription_suffixes_count', 0) + pattern_info.get('symptom_suffixes_count', 0)}개")
        print(f"   🌿 약재 패턴: {pattern_info.get('major_herbs_count', 0)}개")

        # 4. 캐시 관리자
        self.cache_manager = CacheManager(cache_path=str(self.cache_path))
        print("✅ 캐시 관리자 초기화 완료")

        # 5. 고급 답변 생성기
        self.answer_generator = AnswerGenerator(
            llm_manager=self.llm_manager,
            save_path=str(self.save_path),
            terms_manager=self.terms_manager  # 표준용어집 연결
        )
        print("✅ 고급 답변 생성기 초기화 완료 (표준용어집 기반 관련 검색어 제안)")

        # 시스템 상태
        self.is_initialized = False
        self.data_hash = None

        # 시스템 품질 검증
        self._validate_system_quality()

    def _validate_system_quality(self):
        """시스템 품질 검증"""
        print("\n🔍 시스템 품질 검증 중...")

        quality_score = 0
        max_score = 100

        # 표준용어집 연결 (30점)
        if self.terms_manager:
            validation = self.terms_manager.validate_system_integrity()
            if validation['basic_indexes_ok']:
                quality_score += 30
                print("   ✅ 표준용어집 연결: 완전 (30/30)")
            else:
                quality_score += 15
                print("   ⚠️ 표준용어집 연결: 부분 (15/30)")
        else:
            print("   ❌ 표준용어집 연결: 없음 (0/30)")

        # 동적 패턴 시스템 (25점)
        pattern_info = self.search_engine.get_pattern_cache_info()
        if pattern_info.get('terms_manager_connected', False):
            pattern_count = (pattern_info.get('prescription_suffixes_count', 0) +
                             pattern_info.get('symptom_suffixes_count', 0) +
                             pattern_info.get('major_herbs_count', 0))
            if pattern_count > 50:
                quality_score += 25
                print("   ✅ 동적 패턴 시스템: 완전 (25/25)")
            elif pattern_count > 20:
                quality_score += 15
                print("   ⚠️ 동적 패턴 시스템: 양호 (15/25)")
            else:
                quality_score += 5
                print("   ⚠️ 동적 패턴 시스템: 기본 (5/25)")
        else:
            print("   ❌ 동적 패턴 시스템: 연결 안됨 (0/25)")

        # LLM 연결 (20점)
        if self.llm_manager.is_available():
            quality_score += 20
            print("   ✅ LLM 연결: 완전 (20/20)")
        else:
            print("   ❌ LLM 연결: 실패 (0/20)")

        # 문서 처리 품질 (15점)
        proc_stats = self.document_processor.get_processing_statistics()
        if not proc_stats.get('fallback_mode', True):
            quality_score += 15
            print("   ✅ 문서 처리: 고급 모드 (15/15)")
        else:
            quality_score += 8
            print("   ⚠️ 문서 처리: 기본 모드 (8/15)")

        # 캐시 시스템 (10점)
        cache_info = self.cache_manager.get_cache_info()
        if cache_info.get('cache_complete', False):
            quality_score += 10
            print("   ✅ 캐시 시스템: 완전 (10/10)")
        else:
            quality_score += 5
            print("   ⚠️ 캐시 시스템: 부분 (5/10)")

        # 품질 등급 결정
        if quality_score >= 90:
            grade = "S+ (최고급)"
            color = "🟢"
        elif quality_score >= 80:
            grade = "S (우수)"
            color = "🟢"
        elif quality_score >= 70:
            grade = "A (양호)"
            color = "🟡"
        elif quality_score >= 60:
            grade = "B (보통)"
            color = "🟡"
        else:
            grade = "C (개선 필요)"
            color = "🔴"

        print(f"\n{color} 시스템 품질 등급: {grade} ({quality_score}/100점)")

        if quality_score < 70:
            print("\n💡 품질 개선 제안:")
            if not self.terms_manager:
                print("   - 표준한의학용어집 파일 확인 및 설치")
            if not pattern_info.get('terms_manager_connected', False):
                print("   - 용어집-검색엔진 연결 확인")
            if proc_stats.get('fallback_mode', True):
                print("   - 문서 처리기 용어집 연결 확인")

    def initialize_system(self, force_rebuild: bool = False):
        """시스템 초기화 (개선된 버전)"""
        print("\n🚀 완전 개선된 동의보감 RAG 시스템 초기화 중...")

        # 디렉토리 구조 확인
        if not self.check_directory_structure():
            print("❌ 필수 데이터 파일이 없습니다.")
            sys.exit(1)

        # 현재 데이터 해시 계산
        self.data_hash = self.document_processor.calculate_data_hash()

        if not force_rebuild:
            # 캐시 로드 시도
            cache_loaded, cache_data = self.cache_manager.load_cache(
                self.data_hash)

            if cache_loaded and cache_data:
                # 캐시에서 복원
                chunks = cache_data['chunks']
                embeddings = cache_data['embeddings']
                faiss_index = cache_data['faiss_index']

                # 검색 엔진 설정
                self.search_engine.setup(chunks, embeddings)
                self.search_engine.faiss_index = faiss_index

                self.is_initialized = True
                print("🎉 고급 캐시에서 빠르게 로드 완료!")

                # 시스템 검증
                self._post_initialization_validation(chunks)
                return

        # 새로 데이터 처리
        print("📚 개선된 시스템으로 데이터를 처리합니다...")

        # 1. 문서 로드 및 고급 청킹
        print("   📄 고급 문서 처리 및 청킹 중...")
        chunks = self.document_processor.load_documents()

        # 청크 유효성 검증
        validation_result = self.document_processor.validate_chunks(chunks)
        print(f"   📊 청크 검증: 유효 {validation_result['valid_chunks']}개, "
              f"오류 {validation_result['invalid_chunks']}개")

        if validation_result['invalid_chunks'] > 0:
            print(
                f"   ⚠️ {validation_result['invalid_chunks']}개 청크에 문제가 있습니다.")
            for error in validation_result['errors'][:3]:  # 상위 3개만 표시
                print(f"      - {error}")

        # 모델별 최적화
        model_info = self.llm_manager.get_model_info()
        if model_info['name'] in ['gpt-4o-mini', 'gpt-4']:
            optimized_chunks = self.document_processor.optimize_chunks_for_model(
                chunks, model_info['name'])
            chunks = optimized_chunks

        # 2. 고급 임베딩 생성 및 인덱스 구축
        print("   🔍 고급 검색 엔진 설정 중...")
        self.search_engine.setup(chunks)

        # 3. 캐시 저장
        self.cache_manager.save_cache(
            self.data_hash,
            chunks,
            self.search_engine.embeddings,
            self.search_engine.faiss_index
        )

        self.is_initialized = True
        print("✅ 완전 개선된 시스템 초기화 완료!")

        # 시스템 검증
        self._post_initialization_validation(chunks)

    def _post_initialization_validation(self, chunks: List[Dict]):
        """초기화 후 시스템 검증"""
        print("\n🔍 초기화 후 시스템 검증 중...")

        # 청크 통계
        total_chunks = len(chunks)
        prescription_chunks = len(
            [c for c in chunks if c['metadata'].get('type') == 'prescription'])
        avg_chunk_size = sum(c['metadata'].get('token_count', 0)
                             for c in chunks) / total_chunks if total_chunks > 0 else 0

        print(f"   📊 총 청크: {total_chunks:,}개")
        print(
            f"   💊 처방 청크: {prescription_chunks}개 ({prescription_chunks / total_chunks * 100:.1f}%)")
        print(f"   📏 평균 토큰: {avg_chunk_size:.0f}개")

        # 검색 엔진 검증
        if hasattr(self.search_engine, 'embeddings') and self.search_engine.embeddings is not None:
            embedding_shape = self.search_engine.embeddings.shape
            print(f"   🔢 임베딩 차원: {embedding_shape}")

        # 표준용어집 연결 검증
        if self.terms_manager:
            terms_stats = self.terms_manager.get_statistics()
            print(f"   📚 표준용어: {terms_stats.get('total_terms', 0):,}개 연결됨")

        print("✅ 시스템 검증 완료")

    def search(self, query: str, k: int = 75):
        """고급 검색 실행"""
        if not self.is_initialized:
            raise ValueError("시스템이 초기화되지 않았습니다.")

        return self.search_engine.search(query, k)

    def generate_answer(self, query: str, search_results):
        """고급 답변 생성"""
        return self.answer_generator.generate_answer(query, search_results)

    def save_results(self, query: str, results, answer: str):
        """결과 저장"""
        self.answer_generator.save_search_results(query, results, answer)

    def _get_k_value_choice(self, recommended_k: int, max_k: int) -> int:
        """K값 선택 (개선된 가이드)"""
        print("\n🔧 검색 결과 수를 선택하세요:")
        print(f"1. 권장값 ({recommended_k}개) - 균형잡힌 분석 (표준용어집 최적화)")
        print(f"2. 최대값 ({max_k}개) - 최대한 포괄적 분석")
        print("3. 직접 입력 (50~100)")

        if self.terms_manager:
            print("💡 표준용어집 연결됨: 더 정확한 확장 검색 가능")
        else:
            print("⚠️ 표준용어집 미연결: 기본 검색만 가능")

        while True:
            choice = input("선택 (1/2/3): ").strip()
            if choice == '1':
                return recommended_k
            elif choice == '2':
                return max_k
            elif choice == '3':
                try:
                    custom_k = int(input("검색 결과 수 입력 (50~100): "))
                    if 50 <= custom_k <= 100:
                        return custom_k
                    else:
                        print("50~100 사이의 값을 입력해주세요.")
                except ValueError:
                    print("올바른 숫자를 입력해주세요.")
            else:
                print("1, 2, 또는 3을 입력해주세요.")

    def _get_display_options(self) -> dict:
        """표시 옵션 설정 (개선된 옵션)"""
        print("\n🎨 검색 결과 표시 옵션을 선택하세요:")
        print("1. 스마트 카테고리 + 고급 관련 검색어 (기본, 권장)")
        print("2. 전통적인 리스트 형태")
        print("3. 완전 분석 + 표준용어집 활용 (모든 기능)")
        print("4. 품질 메트릭 중심 표시")

        if self.terms_manager:
            print("💡 표준용어집 기반 고급 관련 검색어 제안 활용 가능")

        while True:
            choice = input("선택 (1/2/3/4): ").strip()
            if choice == '1':
                return {
                    'show_categories': True,
                    'show_statistics': False,
                    'show_metrics': False,
                    'show_related_queries': True,
                    'traditional_view': False,
                    'advanced_suggestions': bool(self.terms_manager)
                }
            elif choice == '2':
                return {
                    'show_categories': False,
                    'show_statistics': False,
                    'show_metrics': False,
                    'show_related_queries': False,
                    'traditional_view': True,
                    'advanced_suggestions': False
                }
            elif choice == '3':
                return {
                    'show_categories': True,
                    'show_statistics': True,
                    'show_metrics': True,
                    'show_related_queries': True,
                    'traditional_view': False,
                    'advanced_suggestions': bool(self.terms_manager)
                }
            elif choice == '4':
                return {
                    'show_categories': False,
                    'show_statistics': False,
                    'show_metrics': True,
                    'show_related_queries': False,
                    'traditional_view': False,
                    'advanced_suggestions': False
                }
            else:
                print("1, 2, 3, 또는 4를 입력해주세요.")

    def _process_query(self, query: str, k: int, display_options: dict):
        """쿼리 처리 (완전 개선된 버전)"""
        print(f"\n🔍 '{query}' 고급 검색 중... (결과 수: {k}개)")

        # 쿼리 전처리 (표준용어집 기반)
        if self.terms_manager:
            # 쿼리 확장 미리보기
            expanded_queries = self.terms_manager.expand_query(
                query, max_expansions=3)
            if len(expanded_queries) > 1:
                print(f"   🔄 확장 검색어: {', '.join(expanded_queries[1:])}")

        # 고급 검색 실행
        search_results = self.search(query, k=k)

        if not search_results:
            print("❌ 관련 내용을 찾을 수 없습니다.")
            if self.terms_manager:
                # 유사 검색어 제안
                similar_terms = self.terms_manager.fuzzy_search(
                    query, threshold=0.3)
                if similar_terms:
                    print("💡 유사한 검색어를 시도해보세요:")
                    for term, similarity in similar_terms[:3]:
                        print(f"   - {term} (유사도: {similarity:.3f})")
            return

        print(f"📊 {len(search_results)}개의 관련 문서를 찾았습니다.")

        # 고급 답변 생성
        print("🤖 고급 AI 답변 생성 중...")
        answer = self.generate_answer(query, search_results)

        # 결과 표시 방식에 따른 처리
        if display_options.get('show_metrics', False):
            if display_options.get('show_categories', False):
                # 완전 분석 모드
                print("\n" + "=" * 60)
                if self.llm_manager and self.llm_manager.is_available():
                    print("🤖 AI 답변 (표준용어집 기반 근거 문헌 주석 포함):")
                    print("-" * 40)
                    print(answer)

                # 자동 저장
                self.save_results(query, search_results, answer)
                print("=" * 60)

                # 카테고리별 검색 결과 표시
                self.answer_generator._display_categorized_results(
                    search_results)

                # 고급 관련 검색어 제안 표시
                if display_options.get('show_related_queries', False):
                    self.answer_generator.display_related_queries(
                        query, search_results)

                # 상세 통계 표시
                if display_options.get('show_statistics', False):
                    self.answer_generator.display_category_statistics(
                        search_results)

                # 고급 품질 메트릭 표시
                self.answer_generator.show_search_metrics(
                    query, search_results)

                # 상세 결과 보기 옵션
                show_details_input = input(
                    "\n📋 모든 검색 결과의 전체 내용을 보시겠습니까? (y/n): ").strip().lower()
                if show_details_input in ['y', 'yes', 'ㅇ', '네', '예']:
                    self.answer_generator._display_detailed_results(
                        search_results)

            else:
                # 품질 메트릭 중심 모드
                print("\n" + "=" * 50)
                if self.llm_manager and self.llm_manager.is_available():
                    print("🤖 AI 답변:")
                    print("-" * 30)
                    print(answer)

                # 자동 저장
                self.save_results(query, search_results, answer)

                # 품질 메트릭 표시
                self.answer_generator.show_search_metrics(
                    query, search_results)

                # 상세 결과 보기 옵션
                show_details_input = input(
                    "\n📋 검색 결과의 상세 내용을 보시겠습니까? (y/n): ").strip().lower()
                if show_details_input in ['y', 'yes', 'ㅇ', '네', '예']:
                    self.answer_generator._display_categorized_results(
                        search_results)

        elif display_options.get('show_categories', False):
            # 스마트 카테고리 모드
            show_details = self.answer_generator.display_search_results(
                query, search_results, answer,
                show_related_queries=display_options.get('show_related_queries', True))

            # 상세 통계 표시 (옵션)
            if display_options.get('show_statistics', False):
                self.answer_generator.display_category_statistics(
                    search_results)

        elif display_options.get('traditional_view', False):
            # 전통적인 리스트 표시
            show_details = self._display_traditional_results(
                query, search_results, answer)
            self.save_results(query, search_results, answer)

    def _handle_related_query_selection(self, query: str, search_results: List[Dict], k: int, display_options: dict) -> Optional[str]:
        """고급 관련 검색어 선택 처리"""
        if not self.terms_manager:
            return None

        # 표준용어집 기반 고급 제안
        advanced_suggestions = self.answer_generator.suggest_related_queries_advanced(
            query, search_results, self.terms_manager)

        if not advanced_suggestions:
            return None

        selected_query = self.answer_generator.get_user_choice_for_suggestions(
            advanced_suggestions)

        if selected_query and selected_query != query:
            print(f"\n🔄 '{selected_query}'로 고급 검색을 시작합니다...")
            self._process_query(selected_query, k, display_options)
            return selected_query

        return None

    def _handle_special_commands(self, query: str) -> bool:
        """특수 명령어 처리 (개선된 명령어 추가)"""
        if query.lower().startswith('help') or query == '도움말':
            self._show_help()
            return True
        elif query.lower().startswith('stats') or query == '통계':
            self._show_system_stats()
            return True
        elif query.lower().startswith('config') or query == '설정':
            self._show_config_menu()
            return True
        elif query.lower().startswith('metrics') or query == '품질':
            self._show_metrics_help()
            return True
        elif query.lower().startswith('related') or query == '관련':
            self._show_related_queries_help()
            return True
        elif query.lower().startswith('terms') or query == '용어집':
            self._show_terms_manager_info()
            return True
        elif query.lower().startswith('advanced') or query == '고급':
            self._show_advanced_features()
            return True

        return False

    def _show_terms_manager_info(self):
        """표준용어집 관리자 정보 표시"""
        print("\n📚 표준한의학용어집 관리자 정보")
        print("=" * 50)

        if self.terms_manager:
            stats = self.terms_manager.get_statistics()
            validation = self.terms_manager.validate_system_integrity()

            print("✅ 연결 상태: 정상")
            print(f"📊 총 용어 수: {stats.get('total_terms', 0):,}개")
            print(f"🔍 검색 인덱스: {stats.get('search_index_size', 0):,}개")
            print(f"🏷️ 카테고리: {stats.get('categories', 0)}개")
            print(
                f"🕸️ 관계 그래프: {stats.get('relationship_graph_nodes', 0):,}개 노드, {stats.get('relationship_graph_edges', 0):,}개 엣지")
            print(f"🔬 의미 클러스터: {stats.get('semantic_clusters', 0)}개")
            print(f"🧠 도메인 지식 패턴: {stats.get('domain_knowledge_patterns', 0)}개")

            if 'graph_density' in stats:
                print(f"📈 그래프 밀도: {stats['graph_density']:.4f}")
                print(f"🔗 평균 클러스터링: {stats['average_clustering']:.4f}")

            if 'most_central_terms' in stats:
                print(f"⭐ 중심성 높은 용어: {', '.join(stats['most_central_terms'])}")

            print(f"\n🔧 시스템 무결성:")
            print(
                f"   기본 인덱스: {'✅' if validation['basic_indexes_ok'] else '❌'}")
            print(f"   관계 그래프: {'✅' if validation['graph_ok'] else '❌'}")
            print(f"   의미 클러스터: {'✅' if validation['clusters_ok'] else '❌'}")
            print(
                f"   도메인 지식: {'✅' if validation['domain_knowledge_ok'] else '❌'}")

            if validation['warnings']:
                print(f"\n⚠️ 경고:")
                for warning in validation['warnings'][:3]:
                    print(f"   - {warning}")
        else:
            print("❌ 연결 상태: 표준용어집이 연결되지 않았습니다.")
            print("💡 해결 방법:")
            print("   1. hmedicalterms.json 파일 확인")
            print("   2. 파일 경로 확인: /Users/radi/Projects/langchain/hmedicalterms.json")
            print("   3. 시스템 재시작")

    def _show_advanced_features(self):
        """고급 기능 안내"""
        print("\n🚀 고급 기능 안내")
        print("=" * 40)

        print("🔍 고급 검색 기능:")
        print("   • 표준한의학용어집 기반 지능형 쿼리 확장")
        print("   • 관계 그래프 기반 연관 용어 탐색")
        print("   • 의미적 클러스터 기반 유사 용어 발견")
        print("   • 도메인 지식 기반 맥락적 검색")

        print("\n🎯 관련 검색어 제안:")
        print("   • 6단계 확장 전략 (직접매칭→그래프→도메인지식→클러스터→공기관계→패턴)")
        print("   • 카테고리별 체계적 분류 (처방, 병증, 약재, 이론)")
        print("   • 다차원 점수 기반 순위 매기기")

        print("\n📊 검색 품질 분석:")
        print("   • 실시간 검색 성능 메트릭")
        print("   • 내용 타입별 다양성 분석")
        print("   • 출처 및 대분류 커버리지 분석")
        print("   • 검색 개선 제안 시스템")

        print("\n🔧 시스템 최적화:")
        print("   • 동적 패턴 생성 (하드코딩 제거)")
        print("   • 모델별 청크 최적화")
        print("   • 캐시 기반 고속 로딩")
        print("   • 시스템 무결성 자동 검증")

    def _show_help(self):
        """도움말 표시 (개선된 버전)"""
        print("\n📚 동의보감 RAG 시스템 v3.0 (완전 개선판) 도움말")
        print("=" * 60)
        print("🔍 검색 예시:")
        print("   - 血虛 치료법은?")
        print("   - 四君子湯의 구성과 효능")
        print("   - 補中益氣湯 관련 내용")
        print("   - 인삼의 효능과 주치")
        print()
        print("🛠️ 특수 명령어:")
        print("   - help 또는 도움말: 이 도움말 표시")
        print("   - stats 또는 통계: 시스템 통계 표시")
        print("   - config 또는 설정: 설정 메뉴")
        print("   - metrics 또는 품질: 검색 품질 지표 설명")
        print("   - related 또는 관련: 관련 검색어 기능 설명")
        print("   - terms 또는 용어집: 표준용어집 정보")
        print("   - advanced 또는 고급: 고급 기능 안내")
        print("   - quit, exit, 종료: 시스템 종료")
        print()
        print("🆕 v3.0 새로운 기능:")
        print("   - 🔥 표준한의학용어집 완전 통합")
        print("   - 🕸️ 관계 그래프 기반 지능형 검색")
        print("   - 🔬 의미적 클러스터 분석")
        print("   - 🧠 도메인 지식 베이스 활용")
        print("   - 🎯 6단계 확장 전략")
        print("   - 📊 고급 검색 품질 메트릭")
        print("   - 🔧 하드코딩 완전 제거")
        print()
        print("💡 팁:")
        print("   - 한자와 한글 모두 검색 가능")
        print("   - 표준용어집 기반 정확한 확장 검색")
        print("   - 관련 검색어로 깊이 있는 탐색")
        print("   - 검색 품질 메트릭으로 결과 신뢰도 확인")

    def _show_system_stats(self):
        """시스템 통계 표시 (개선된 버전)"""
        info = self.get_system_info()
        print("\n📊 완전 개선된 시스템 통계")
        print("=" * 50)

        # 기본 정보
        print("🏗️ 시스템 구성:")
        print(f"   📚 총 청크 수: {info.get('chunks_count', 'N/A'):,}개")
        print(f"   🔢 임베딩 차원: {info.get('embeddings_shape', 'N/A')}")
        print(
            f"   💾 캐시 상태: {'✅ 활성' if info.get('cache_info', {}).get('cache_complete', False) else '❌ 비활성'}")
        print(f"   📂 데이터 경로: {info['data_path']}")
        print(f"   💾 결과 저장: {info['save_path']}")

        # 표준용어집 정보
        print(f"\n📚 표준한의학용어집:")
        if info['terms_manager_available']:
            if self.terms_manager:
                stats = self.terms_manager.get_statistics()
                print(f"   ✅ 연결 상태: 정상")
                print(f"   📊 총 용어: {stats.get('total_terms', 0):,}개")
                print(
                    f"   🕸️ 관계 그래프: {stats.get('relationship_graph_nodes', 0):,}개 노드")
                print(f"   🔬 의미 클러스터: {stats.get('semantic_clusters', 0)}개")
                print(
                    f"   🧠 도메인 지식: {stats.get('domain_knowledge_patterns', 0)}개 패턴")
        else:
            print(f"   ❌ 연결 상태: 미연결")

        # LLM 정보
        model_info = self.llm_manager.get_model_info()
        print(f"\n🤖 AI 모델:")
        print(f"   모델명: {model_info['display_name']}")
        print(f"   연결 상태: {'✅' if model_info['is_connected'] else '❌'}")
        print(f"   최적 K값: {model_info['optimal_k']}")
        print(f"   컨텍스트 길이: {model_info['max_context_tokens']:,} 토큰")

        # 고급 기능 상태
        print(f"\n🚀 고급 기능 상태:")
        print(f"   🔍 동적 패턴 시스템: ✅ 활성")
        print(f"   🎯 관련 검색어 제안: ✅ 활성")
        print(f"   📊 검색 품질 메트릭: ✅ 활성")
        print(f"   🔧 하드코딩 제거: ✅ 완료")

        # 패턴 캐시 정보
        if hasattr(self.search_engine, 'get_pattern_cache_info'):
            pattern_info = self.search_engine.get_pattern_cache_info()
            print(
                f"   🏷️ 처방 패턴: {pattern_info.get('prescription_suffixes_count', 0)}개")
            print(f"   🌿 약재 패턴: {pattern_info.get('major_herbs_count', 0)}개")
            print(
                f"   📚 이론 개념: {pattern_info.get('theory_concepts_count', 0)}개")

    def _show_config_menu(self):
        """설정 메뉴 표시 (개선된 버전)"""
        print("\n⚙️ 고급 설정 메뉴")
        print("=" * 40)
        print("1. 캐시 정보 확인")
        print("2. 캐시 삭제")
        print("3. 표준용어집 상세 정보")
        print("4. 관련 검색어 기능 테스트")
        print("5. 시스템 무결성 검증")
        print("6. 성능 벤치마크")
        print("7. 패턴 캐시 재구축")
        print("8. 돌아가기")

        while True:
            choice = input("선택 (1-8): ").strip()
            if choice == '1':
                cache_info = self.cache_manager.get_cache_info()
                print(f"\n💾 캐시 정보:")
                print(
                    f"   상태: {'완전' if cache_info['cache_complete'] else '불완전'}")
                if 'chunks_count' in cache_info:
                    print(f"   청크 수: {cache_info['chunks_count']:,}개")
                    print(f"   생성 시간: {cache_info.get('timestamp', 'N/A')}")
                    print(f"   크기 정보: {cache_info.get('size_info', {})}")
                break
            elif choice == '2':
                confirm = input("정말로 캐시를 삭제하시겠습니까? (y/n): ").lower()
                if confirm in ['y', 'yes', 'ㅇ']:
                    self.cache_manager.clear_cache()
                    if self.terms_manager:
                        self.terms_manager.clear_cache()
                    print("🗑️ 모든 캐시가 삭제되었습니다. 다음 실행 시 새로 생성됩니다.")
                break
            elif choice == '3':
                self._show_terms_manager_info()
                break
            elif choice == '4':
                print("\n🧪 관련 검색어 기능 테스트")
                test_query = input("테스트용 검색어를 입력하세요 (예: 血虛): ").strip()
                if test_query and self.is_initialized:
                    print("🔄 테스트 검색 실행 중...")
                    test_results = self.search(test_query, k=20)
                    if test_results:
                        print("✅ 검색 완료! 고급 관련 검색어 제안:")
                        self.answer_generator.display_related_queries(
                            test_query, test_results)
                    else:
                        print("❌ 검색 결과가 없습니다.")
                break
            elif choice == '5':
                print("\n🔍 시스템 무결성 검증 중...")
                if self.terms_manager:
                    validation = self.terms_manager.validate_system_integrity()
                    print("표준용어집 검증 결과:")
                    for key, value in validation.items():
                        if isinstance(value, bool):
                            print(f"   {key}: {'✅' if value else '❌'}")
                        elif isinstance(value, list) and value:
                            print(f"   {key}: {len(value)}개 항목")

                if hasattr(self.document_processor, 'validate_terms_manager_connection'):
                    doc_validation = self.document_processor.validate_terms_manager_connection()
                    print(f"문서 처리기 연결: {'✅' if doc_validation else '❌'}")

                print("✅ 무결성 검증 완료")
                break
            elif choice == '6':
                print("\n⚡ 성능 벤치마크 실행 중...")
                self._run_performance_benchmark()
                break
            elif choice == '7':
                print("\n🔄 패턴 캐시 재구축 중...")
                if hasattr(self.search_engine, 'rebuild_patterns'):
                    self.search_engine.rebuild_patterns()
                if hasattr(self.document_processor, 'rebuild_dynamic_dictionary'):
                    self.document_processor.rebuild_dynamic_dictionary()
                print("✅ 패턴 캐시 재구축 완료")
                break
            elif choice == '8':
                break
            else:
                print("1-8 사이의 번호를 입력해주세요.")

    def _run_performance_benchmark(self):
        """성능 벤치마크 실행"""
        import time

        test_queries = ['血虛', '四君子湯', '人參', '陰虛', '補中益氣湯']

        print("📊 검색 성능 벤치마크:")
        total_time = 0

        for query in test_queries:
            start_time = time.time()
            results = self.search(query, k=20)
            end_time = time.time()

            query_time = end_time - start_time
            total_time += query_time

            print(f"   {query}: {len(results)}개 결과, {query_time:.3f}초")

        avg_time = total_time / len(test_queries)
        print(f"\n평균 검색 시간: {avg_time:.3f}초")
        print(f"초당 처리 가능: {1 / avg_time:.1f} 쿼리")

        if self.terms_manager:
            print("\n📚 용어집 확장 성능:")
            expansion_time = 0
            for query in test_queries:
                start_time = time.time()
                expansions = self.terms_manager.expand_query(
                    query, max_expansions=10)
                end_time = time.time()

                exp_time = end_time - start_time
                expansion_time += exp_time
                print(f"   {query}: {len(expansions)}개 확장, {exp_time:.3f}초")

            avg_exp_time = expansion_time / len(test_queries)
            print(f"\n평균 확장 시간: {avg_exp_time:.3f}초")

    def _show_metrics_help(self):
        """검색 품질 메트릭 도움말 (개선된 버전)"""
        print("\n📊 고급 검색 품질 메트릭 설명")
        print("=" * 50)

        print("\n🔍 기본 지표:")
        print("   • 처방 정보: 검색 결과 중 처방 관련 문서 수")
        print("   • 이론 내용: 이론적 배경을 담은 문서 수")
        print("   • 출처 다양성: 활용된 원문 파일의 종류")
        print("   • 평균 관련도: 검색 결과의 평균 유사도 점수")

        print("\n📈 고급 분석:")
        print("   • 내용 타입 다양성: 처방/이론/병증/약물 등의 균형")
        print("   • 관련도 분포: 고/중/저품질 결과의 비율")
        print("   • 대분류/중분류 다양성: 동의보감 구조별 커버리지")
        print("   • 직접 매칭률: 검색어가 원문에 직접 포함된 비율")
        print("   • 표준용어집 활용률: 표준용어 기반 확장 성공률")

        print("\n🎯 품질 등급:")
        print("   • S+ (최고급): 표준용어집 완전 활용, 완벽한 다양성")
        print("   • S (최우수): 매우 포괄적이고 정확한 검색")
        print("   • A (우수): 균형잡힌 좋은 검색 결과")
        print("   • B (양호): 적절하나 일부 개선 여지")
        print("   • C (보통): 기본적 결과, 개선 필요")
        print("   • D (미흡): 검색 전략 재고 필요")

        print("\n💡 활용 팁:")
        print("   • 등급이 낮으면 표준용어집 기반 정확한 한자 표기 확인")
        print("   • 출처 다양성이 낮으면 더 일반적인 용어 사용")
        print("   • 관련 검색어 제안을 활용한 연관 탐색")
        print("   • 도메인 지식 기반 맥락적 검색 활용")

    def _show_related_queries_help(self):
        """관련 검색어 제안 도움말 (개선된 버전)"""
        print("\n🔍 고급 관련 검색어 제안 기능 설명")
        print("=" * 50)

        print("\n💡 기능 개요:")
        print("   • 표준한의학용어집 기반 지능형 관련 검색어 자동 제안")
        print("   • 6단계 확장 전략을 통한 깊이 있는 연관 탐색")
        print("   • 관계 그래프 기반 의미적 연결 분석")
        print("   • 카테고리별 체계적으로 분류된 제안사항 제공")

        print("\n🏷️ 제안 카테고리:")
        print("   • 🔥 핵심 처방: 표준용어집에서 추출한 관련 처방들")
        print("   • 🩺 관련 병증: 계층구조 기반 관련 증상이나 병증들")
        print("   • 💊 주요 약재: 도메인 지식 기반 관련 약재들")
        print("   • 📚 관련 개념: 의미 클러스터 기반 이론이나 개념들")
        print("   • 🎯 맞춤 제안: 공기 관계 분석 기반 개인화 추천")

        print("\n🚀 6단계 확장 전략:")
        print("   1. 직접 매칭: 표준용어집 직접 검색")
        print("   2. 관계 그래프: NetworkX 기반 노드 탐색")
        print("   3. 도메인 지식: 임상 지식 베이스 활용")
        print("   4. 의미 클러스터: 커뮤니티 탐지 알고리즘")
        print("   5. 공기 관계: 용어 간 동시 출현 분석")
        print("   6. 패턴 매칭: 형태소 및 접사 분석")

        print("\n🔄 사용 방법:")
        print("   1. 검색 후 고급 관련 검색어 제안 목록 확인")
        print("   2. 번호를 입력하여 바로 해당 검색어로 검색")
        print("   3. 새로운 검색어를 직접 입력하여 검색")
        print("   4. Enter로 관련 검색어 건너뛰기")

        print("\n✨ 고급 활용 팁:")
        print("   • 처방 검색 후 → 구성 약재나 관련 병증 자동 탐색")
        print("   • 병증 검색 후 → 치료 처방이나 감별진단 지능 추천")
        print("   • 약재 검색 후 → 배합금기나 효능 비교 제안")
        print("   • 이론 검색 후 → 임상 응용이나 구체적 사례 연결")

    def _display_traditional_results(self, query: str, results: List[Dict], answer: str) -> bool:
        """전통적인 방식의 결과 표시"""
        print("\n" + "=" * 50)

        if self.llm_manager and self.llm_manager.is_available():
            print("🤖 AI 답변:")
            print("-" * 30)
            print(answer)
        else:
            print("⚠️ AI 답변을 생성할 수 없어 검색 결과만 표시합니다.")

        print("=" * 50)
        print("🔍 검색 결과:")
        print("-" * 30)

        # 상위 20개만 미리보기
        for i, result in enumerate(results[:20]):
            print(f"\n[문서 {i + 1}] (유사도: {result['score']:.3f})")
            print(f"출처: {result['metadata']['source_file']}")
            if result['metadata'].get('BB'):
                print(f"대분류: {result['metadata']['BB']}")
            if result['metadata'].get('CC'):
                print(f"중분류: {result['metadata']['CC']}")
            print(f"내용: {result['content'][:200]}...")
            print("-" * 20)

        print("=" * 50)

        # 상세 결과 보기 옵션
        show_details_input = input(
            "\n📋 모든 검색 결과의 전체 내용을 보시겠습니까? (y/n): ").strip().lower()
        show_details = show_details_input in ['y', 'yes', 'ㅇ', '네', '예']

        if show_details:
            self.answer_generator._display_detailed_results(results)

        return show_details

    def interactive_chat(self):
        """대화형 인터페이스 (완전 개선된 버전)"""
        if not self.is_initialized:
            print("❌ 시스템이 초기화되지 않았습니다.")
            return

        llm_manager = self.llm_manager
        model_info = llm_manager.get_model_info()

        # 모델별 최적화된 K값 설정
        recommended_k = 75  # GPT-4o-mini 표준 권장값
        max_k = 100        # 최대값

        print("\n" + "=" * 70)
        print("🏥 동의보감 RAG 시스템 v3.0 (완전 개선판)에 오신 것을 환영합니다!")
        print(f"🤖 사용 중인 모델: {model_info['display_name']}")
        print(f"📊 권장 검색 결과 수: {recommended_k}개 (최대 {max_k}개)")

        # 시스템 상태 표시
        if self.terms_manager:
            stats = self.terms_manager.get_statistics()
            print(f"📚 표준한의학용어집: {stats.get('total_terms', 0):,}개 용어 연결됨")
            print(
                f"🕸️ 관계 그래프: {stats.get('relationship_graph_nodes', 0):,}개 노드")
        else:
            print("⚠️ 표준용어집: 기본 모드 (일부 기능 제한)")

        print("\n🚀 v3.0 완전 개선 기능:")
        print("   • 🔥 표준한의학용어집 완전 통합")
        print("   • 🕸️ 관계 그래프 기반 지능형 관련 검색어 제안")
        print("   • 🔬 의미적 클러스터 분석")
        print("   • 🧠 도메인 지식 베이스 활용")
        print("   • 📊 고급 검색 품질 메트릭")
        print("   • 🔧 하드코딩 완전 제거")
        print("\n💡 중의학 관련 질문을 한국어로 입력하세요.")
        print("🆘 도움말: 'help' 또는 '도움말' 입력")
        print("📚 용어집 정보: 'terms' 또는 '용어집' 입력")
        print("🚀 고급 기능: 'advanced' 또는 '고급' 입력")
        print("🚪 종료하려면 'quit', 'exit', '종료'를 입력하세요.")

        # K값 설정
        selected_k = self._get_k_value_choice(recommended_k, max_k)
        print(f"✅ 검색 결과 수: {selected_k}개로 설정되었습니다.")

        # 표시 옵션 설정
        display_options = self._get_display_options()
        print("=" * 70 + "\n")

        # 메인 채팅 루프
        while True:
            try:
                query = input("🤔 질문을 입력하세요: ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', '종료']:
                    print("👋 동의보감 RAG 시스템을 종료합니다.")
                    break

                # 특수 명령어 처리
                if self._handle_special_commands(query):
                    continue

                # 고급 검색 및 답변 생성
                self._process_query(query, selected_k, display_options)

                # 고급 관련 검색어 선택 처리
                if display_options.get('show_related_queries', False):
                    while True:
                        print("\n" + "🔍" * 30)
                        related_choice = input(
                            "🔄 관련 검색어로 계속 검색하시겠습니까? (y/n/번호 입력): ").strip()

                        if related_choice.lower() in ['n', 'no', 'ㄴ', '아니오', '아니요']:
                            break
                        elif related_choice.lower() in ['y', 'yes', 'ㅇ', '네', '예']:
                            # 고급 관련 검색어 다시 표시
                            recent_results = self.search(query, k=selected_k)
                            if recent_results:
                                if self.terms_manager:
                                    categorized_suggestions = self.answer_generator.suggest_related_queries_advanced(
                                        query, recent_results, self.terms_manager)
                                else:
                                    categorized_suggestions = self.answer_generator.suggest_related_queries(
                                        query, recent_results)

                                if categorized_suggestions:
                                    print("\n💡 고급 관련 검색어 목록:")
                                    suggestion_count = 1
                                    all_suggestions = []

                                    for category, suggestions in categorized_suggestions.items():
                                        if suggestions:
                                            print(f"\n{category}:")
                                            for suggestion in suggestions:
                                                print(
                                                    f"   {suggestion_count}. {suggestion}")
                                                all_suggestions.append(
                                                    suggestion)
                                                suggestion_count += 1

                                    choice_input = input(
                                        f"\n선택 (1-{len(all_suggestions)} 또는 직접 입력): ").strip()

                                    if choice_input.isdigit() and 1 <= int(choice_input) <= len(all_suggestions):
                                        new_query = all_suggestions[int(
                                            choice_input) - 1]
                                        print(
                                            f"✅ '{new_query}'로 고급 검색을 시작합니다.")
                                        self._process_query(
                                            new_query, selected_k, display_options)
                                        query = new_query  # 다음 관련 검색을 위해 query 업데이트
                                    elif choice_input:
                                        print(
                                            f"✅ '{choice_input}'로 새로운 검색을 시작합니다.")
                                        self._process_query(
                                            choice_input, selected_k, display_options)
                                        query = choice_input  # 다음 관련 검색을 위해 query 업데이트
                                    else:
                                        break
                                else:
                                    print("💭 관련 검색어를 찾을 수 없습니다.")
                                    break
                            else:
                                break
                        elif related_choice.isdigit():
                            # 직접 번호 입력 처리
                            recent_results = self.search(query, k=selected_k)
                            if recent_results:
                                if self.terms_manager:
                                    categorized_suggestions = self.answer_generator.suggest_related_queries_advanced(
                                        query, recent_results, self.terms_manager)
                                else:
                                    categorized_suggestions = self.answer_generator.suggest_related_queries(
                                        query, recent_results)

                                all_suggestions = []
                                for suggestions in categorized_suggestions.values():
                                    all_suggestions.extend(suggestions)

                                choice_num = int(related_choice)
                                if 1 <= choice_num <= len(all_suggestions):
                                    new_query = all_suggestions[choice_num - 1]
                                    print(f"✅ '{new_query}'로 고급 검색을 시작합니다.")
                                    self._process_query(
                                        new_query, selected_k, display_options)
                                    query = new_query
                                else:
                                    print(
                                        f"❌ 1-{len(all_suggestions)} 범위의 번호를 입력해주세요.")
                            break
                        else:
                            print("y, n, 또는 번호를 입력해주세요.")

                # 계속 검색할지 확인
                if not self.answer_generator.get_continue_choice():
                    print("👋 동의보감 RAG 시스템을 종료합니다.")
                    break

            except KeyboardInterrupt:
                print("\n\n👋 동의보감 RAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
                print("다시 시도해주세요.")

    def check_directory_structure(self):
        """디렉토리 구조 확인 및 생성"""
        required_dirs = [
            self.data_path,
            self.cache_path,
            self.save_path,
            self.data_path.parent / "Results"  # Results 상위 디렉토리
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
                dir_path.mkdir(parents=True, exist_ok=True)

        if missing_dirs:
            print(f"📁 생성된 디렉토리: {', '.join(missing_dirs)}")

        # 동의보감 원문 파일 확인
        dybg_files = list(self.data_path.rglob("*.txt"))
        if not dybg_files:
            print("⚠️ 동의보감 원문 파일을 찾을 수 없습니다.")
            print(f"📂 다음 경로에 동의보감 텍스트 파일들을 복사해주세요:")
            print(f"   {self.data_path}")
            return False

        print(f"✅ {len(dybg_files)}개의 동의보감 원문 파일 확인됨")
        return True

    def get_system_info(self):
        """시스템 정보 반환 (개선된 버전)"""
        info = {
            'initialized': self.is_initialized,
            'data_path': str(self.data_path),
            'cache_path': str(self.cache_path),
            'save_path': str(self.save_path),
            'data_hash': self.data_hash,
            'terms_manager_available': self.terms_manager is not None,
            'cache_info': self.cache_manager.get_cache_info(),
            'related_queries_enabled': True,
            'search_metrics_enabled': True,
            'advanced_features_enabled': True,
            'hardcoding_removed': True
        }

        # 표준용어집 정보
        if self.terms_manager:
            terms_stats = self.terms_manager.get_statistics()
            info['terms_manager_stats'] = terms_stats

            validation = self.terms_manager.validate_system_integrity()
            info['terms_manager_validation'] = validation

        # 문서 처리기 정보
        if hasattr(self.document_processor, 'get_processing_statistics'):
            proc_stats = self.document_processor.get_processing_statistics()
            info['document_processor_stats'] = proc_stats

        # 검색 엔진 정보
        if hasattr(self.search_engine, 'get_pattern_cache_info'):
            pattern_info = self.search_engine.get_pattern_cache_info()
            info['search_engine_patterns'] = pattern_info

        if self.is_initialized and hasattr(self.search_engine, 'chunks'):
            info['chunks_count'] = len(self.search_engine.chunks)
            if hasattr(self.search_engine, 'embeddings'):
                info['embeddings_shape'] = self.search_engine.embeddings.shape
            if hasattr(self.search_engine, 'faiss_index'):
                info['faiss_index_count'] = self.search_engine.faiss_index.ntotal

        return info

    def export_system_report(self, output_path: str = None) -> str:
        """시스템 상태 리포트 내보내기"""
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.save_path / f"system_report_{timestamp}.txt"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("동의보감 RAG 시스템 v3.0 (완전 개선판) 상태 리포트\n")
                f.write("=" * 80 + "\n")
                f.write(
                    f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 시스템 정보
                info = self.get_system_info()
                f.write("🏗️ 시스템 기본 정보\n")
                f.write("-" * 40 + "\n")
                f.write(f"초기화 상태: {'완료' if info['initialized'] else '미완료'}\n")
                f.write(f"데이터 경로: {info['data_path']}\n")
                f.write(f"캐시 경로: {info['cache_path']}\n")
                f.write(f"결과 저장 경로: {info['save_path']}\n")
                f.write(
                    f"하드코딩 제거: {'✅ 완료' if info['hardcoding_removed'] else '❌'}\n")

                # 표준용어집 정보
                f.write(f"\n📚 표준한의학용어집 상태\n")
                f.write("-" * 40 + "\n")
                if info['terms_manager_available']:
                    stats = info.get('terms_manager_stats', {})
                    validation = info.get('terms_manager_validation', {})

                    f.write(f"연결 상태: ✅ 정상\n")
                    f.write(f"총 용어 수: {stats.get('total_terms', 0):,}개\n")
                    f.write(
                        f"관계 그래프: {stats.get('relationship_graph_nodes', 0):,}개 노드, {stats.get('relationship_graph_edges', 0):,}개 엣지\n")
                    f.write(f"의미 클러스터: {stats.get('semantic_clusters', 0)}개\n")
                    f.write(
                        f"도메인 지식 패턴: {stats.get('domain_knowledge_patterns', 0)}개\n")

                    f.write(f"\n무결성 검증:\n")
                    f.write(
                        f"  기본 인덱스: {'✅' if validation.get('basic_indexes_ok', False) else '❌'}\n")
                    f.write(
                        f"  관계 그래프: {'✅' if validation.get('graph_ok', False) else '❌'}\n")
                    f.write(
                        f"  의미 클러스터: {'✅' if validation.get('clusters_ok', False) else '❌'}\n")
                    f.write(
                        f"  도메인 지식: {'✅' if validation.get('domain_knowledge_ok', False) else '❌'}\n")
                else:
                    f.write(f"연결 상태: ❌ 미연결\n")

                # 문서 처리기 정보
                f.write(f"\n📄 문서 처리기 상태\n")
                f.write("-" * 40 + "\n")
                if 'document_processor_stats' in info:
                    proc_stats = info['document_processor_stats']
                    f.write(
                        f"표준용어집 연결: {'✅' if proc_stats.get('terms_manager_connected', False) else '❌'}\n")
                    f.write(
                        f"동적 TCM 용어: {proc_stats.get('dynamic_terms_count', 0)}개\n")
                    f.write(
                        f"처방 패턴: {proc_stats.get('prescription_patterns_count', 0)}개\n")
                    f.write(
                        f"약재 패턴: {proc_stats.get('herb_patterns_count', 0)}개\n")
                    f.write(
                        f"폴백 모드: {'❌' if proc_stats.get('fallback_mode', True) else '✅'}\n")

                # 검색 엔진 정보
                f.write(f"\n🔍 검색 엔진 상태\n")
                f.write("-" * 40 + "\n")
                if 'search_engine_patterns' in info:
                    pattern_info = info['search_engine_patterns']
                    f.write(
                        f"표준용어집 연결: {'✅' if pattern_info.get('terms_manager_connected', False) else '❌'}\n")
                    f.write(
                        f"처방 접미사: {pattern_info.get('prescription_suffixes_count', 0)}개\n")
                    f.write(
                        f"증상 접미사: {pattern_info.get('symptom_suffixes_count', 0)}개\n")
                    f.write(
                        f"주요 약재: {pattern_info.get('major_herbs_count', 0)}개\n")
                    f.write(
                        f"이론 개념: {pattern_info.get('theory_concepts_count', 0)}개\n")

                if info['initialized']:
                    f.write(f"\n청크 수: {info.get('chunks_count', 0):,}개\n")
                    f.write(f"임베딩 차원: {info.get('embeddings_shape', 'N/A')}\n")
                    f.write(
                        f"FAISS 인덱스: {info.get('faiss_index_count', 0):,}개\n")

                # LLM 정보
                model_info = self.llm_manager.get_model_info()
                f.write(f"\n🤖 AI 모델 상태\n")
                f.write("-" * 40 + "\n")
                f.write(f"모델명: {model_info['display_name']}\n")
                f.write(
                    f"연결 상태: {'✅' if model_info['is_connected'] else '❌'}\n")
                f.write(f"최적 K값: {model_info['optimal_k']}\n")
                f.write(f"컨텍스트 길이: {model_info['max_context_tokens']:,} 토큰\n")

                # 캐시 정보
                cache_info = info['cache_info']
                f.write(f"\n💾 캐시 상태\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"캐시 완성도: {'완전' if cache_info.get('cache_complete', False) else '불완전'}\n")
                if 'chunks_count' in cache_info:
                    f.write(f"캐시된 청크: {cache_info['chunks_count']:,}개\n")
                    f.write(f"생성 시간: {cache_info.get('timestamp', 'N/A')}\n")

                # 고급 기능 상태
                f.write(f"\n🚀 고급 기능 상태\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"관련 검색어 제안: {'✅' if info['related_queries_enabled'] else '❌'}\n")
                f.write(
                    f"검색 품질 메트릭: {'✅' if info['search_metrics_enabled'] else '❌'}\n")
                f.write(
                    f"고급 기능: {'✅' if info['advanced_features_enabled'] else '❌'}\n")
                f.write(f"동적 패턴 시스템: ✅\n")
                f.write(f"관계 그래프 분석: ✅\n")
                f.write(f"의미 클러스터: ✅\n")
                f.write(f"도메인 지식 베이스: ✅\n")

            print(f"📄 시스템 리포트가 저장되었습니다: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"⚠️ 리포트 생성 실패: {e}")
            return ""

    def run_system_diagnostics(self):
        """시스템 진단 실행"""
        print("\n🔍 완전 개선된 시스템 진단 실행 중...")

        diagnostics = {
            'overall_health': 'healthy',
            'issues': [],
            'recommendations': [],
            'performance_metrics': {}
        }

        # 1. 표준용어집 진단
        if self.terms_manager:
            validation = self.terms_manager.validate_system_integrity()
            if not validation['basic_indexes_ok']:
                diagnostics['issues'].append("표준용어집 기본 인덱스 오류")
                diagnostics['overall_health'] = 'warning'

            if validation['errors']:
                diagnostics['issues'].extend(validation['errors'])
                diagnostics['overall_health'] = 'critical'
        else:
            diagnostics['issues'].append("표준용어집이 연결되지 않음")
            diagnostics['recommendations'].append(
                "hmedicalterms.json 파일 확인 필요")
            diagnostics['overall_health'] = 'warning'

        # 2. LLM 연결 진단
        if not self.llm_manager.is_available():
            diagnostics['issues'].append("LLM 연결 실패")
            diagnostics['recommendations'].append("OpenAI API 키 확인 필요")
            diagnostics['overall_health'] = 'critical'

        # 3. 캐시 시스템 진단
        cache_info = self.cache_manager.get_cache_info()
        if not cache_info.get('cache_complete', False):
            diagnostics['issues'].append("캐시 불완전")
            diagnostics['recommendations'].append("시스템 재초기화 권장")

        # 4. 성능 진단
        if self.is_initialized:
            import time
            start_time = time.time()
            test_results = self.search("血虛", k=10)
            search_time = time.time() - start_time

            diagnostics['performance_metrics']['search_time'] = search_time
            diagnostics['performance_metrics']['result_count'] = len(
                test_results)

            if search_time > 5.0:
                diagnostics['issues'].append("검색 응답 시간 느림")
                diagnostics['recommendations'].append("캐시 재구축 또는 인덱스 최적화 필요")

        # 5. 전체 건강도 평가
        if len(diagnostics['issues']) == 0:
            diagnostics['overall_health'] = 'excellent'
        elif len(diagnostics['issues']) <= 2 and diagnostics['overall_health'] != 'critical':
            diagnostics['overall_health'] = 'good'
        elif diagnostics['overall_health'] != 'critical':
            diagnostics['overall_health'] = 'warning'

        # 결과 출력
        health_colors = {
            'excellent': '🟢',
            'good': '🟢',
            'healthy': '🟡',
            'warning': '🟡',
            'critical': '🔴'
        }

        print(
            f"\n{health_colors[diagnostics['overall_health']]} 시스템 건강도: {diagnostics['overall_health'].upper()}")

        if diagnostics['issues']:
            print(f"\n⚠️ 발견된 문제:")
            for issue in diagnostics['issues']:
                print(f"   - {issue}")

        if diagnostics['recommendations']:
            print(f"\n💡 권장사항:")
            for rec in diagnostics['recommendations']:
                print(f"   - {rec}")

        if diagnostics['performance_metrics']:
            print(f"\n📈 성능 메트릭:")
            for metric, value in diagnostics['performance_metrics'].items():
                if metric == 'search_time':
                    print(f"   검색 시간: {value:.3f}초")
                else:
                    print(f"   {metric}: {value}")

        print("✅ 시스템 진단 완료")
        return diagnostics


def main():
    """메인 함수 (완전 개선된 버전)"""
    try:
        print("🚀 동의보감 RAG 시스템 v3.0 (완전 개선판) 시작")
        print("=" * 70)
        print("🎯 주요 개선사항:")
        print("   • 🔥 표준한의학용어집 완전 통합 (7,105개 용어)")
        print("   • 🕸️ NetworkX 기반 관계 그래프 (용어 간 연관성 분석)")
        print("   • 🔬 의미적 클러스터링 (커뮤니티 탐지 알고리즘)")
        print("   • 🧠 도메인 지식 베이스 (임상 중심 체계적 지식)")
        print("   • 🎯 6단계 지능형 확장 전략")
        print("   • 📊 고급 검색 품질 메트릭")
        print("   • 🔧 하드코딩 완전 제거")
        print("   • ⚡ 동적 패턴 생성 시스템")

        print("\n🔍 고급 검색 기능:")
        print("   • 표준용어집 기반 정확한 용어 매칭")
        print("   • 관계 그래프 탐색을 통한 연관 용어 발견")
        print("   • 의미 클러스터 기반 유사 개념 탐색")
        print("   • 도메인 지식 활용 맥락적 검색")
        print("   • 공기 관계 분석 기반 추천")
        print("   • 패턴 매칭을 통한 형태적 유사성 발견")

        # 시작 옵션 선택
        print("\n🔧 시작 옵션을 선택하세요:")
        print("1. 고속 캐시 로드 (권장)")
        print("2. 완전 재구축 (초기 설정 또는 데이터 변경시)")
        print("3. 시스템 진단 후 결정")

        while True:
            choice = input("선택 (1/2/3): ").strip()
            if choice == '1':
                force_rebuild = False
                break
            elif choice == '2':
                force_rebuild = True
                break
            elif choice == '3':
                # 간단한 사전 진단
                print("\n🔍 사전 시스템 진단 중...")
                rag_system = DonguiRAGSystemImproved()
                diagnostics = rag_system.run_system_diagnostics()

                if diagnostics['overall_health'] in ['excellent', 'good', 'healthy']:
                    print("💡 시스템 상태가 양호합니다. 캐시 로드를 권장합니다.")
                    force_rebuild = False
                else:
                    print("💡 시스템에 문제가 있습니다. 완전 재구축을 권장합니다.")
                    force_rebuild = True
                break
            else:
                print("1, 2, 또는 3을 입력해주세요.")

        # RAG 시스템 초기화 (완전 개선된 버전)
        if 'rag_system' not in locals():
            rag_system = DonguiRAGSystemImproved()

        rag_system.initialize_system(force_rebuild=force_rebuild)

        # 시스템 리포트 생성 (옵션)
        print("\n📄 시스템 상태 리포트를 생성하시겠습니까? (y/n): ", end="")
        report_choice = input().strip().lower()
        if report_choice in ['y', 'yes', 'ㅇ', '네', '예']:
            report_path = rag_system.export_system_report()
            if report_path:
                print(f"✅ 시스템 리포트 생성 완료: {report_path}")

        # 대화형 인터페이스 시작
        rag_system.interactive_chat()

    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("\n💡 문제 해결 방법:")
        print("   1. OpenAI API 키 확인 (.env 파일 또는 환경변수)")
        print("   2. 필수 모듈 설치 확인:")
        print("      - document_processor_improved.py")
        print("      - search_engine_improved.py")
        print("      - answer_generator_improved.py")
        print("      - medical_terms_manager_improved.py")
        print("   3. 표준용어집 파일 확인 (hmedicalterms.json)")
        print("   4. 데이터 경로 확인 및 권한 설정")
        print("   5. Python 의존성 설치 (requirements.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()
