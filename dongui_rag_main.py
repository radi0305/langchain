#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의보감 RAG 시스템 메인 모듈 - dongui_rag_main.py (관련 검색어 제안 통합 버전)
검색 결과 그룹핑, 검색 품질 메트릭 및 관련 검색어 제안을 포함한 통합 실행 모듈
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")

# 모듈 임포트
try:
    from document_processor import DocumentProcessor
    from search_engine import SearchEngine
    from cache_manager import CacheManager
    from answer_generator import AnswerGenerator
    from medical_terms_manager import MedicalTermsManager
    from llm_manager import LLMManager
except ImportError as e:
    print(f"필수 모듈이 없습니다: {e}")
    print("다음 파일들이 같은 디렉토리에 있는지 확인해주세요:")
    print("- document_processor.py")
    print("- search_engine.py")
    print("- cache_manager.py")
    print("- answer_generator.py")
    print("- medical_terms_manager.py")
    print("- llm_manager.py")
    sys.exit(1)


class DonguiRAGSystem:
    def __init__(self,
                 data_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG",
                 cache_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG/cache",
                 save_path: str = "/Users/radi/Projects/langchainDATA/Results/DYBGsearch"):
        """동의보감 RAG 시스템 초기화 (데이터 디렉토리 분리)"""
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

        # 각 모듈 초기화
        print("🔧 시스템 모듈 초기화 중...")

        # 1. 표준용어집 관리자
        try:
            self.terms_manager = MedicalTermsManager()
            print("✅ 표준용어집 관리자 초기화 완료")
        except Exception as e:
            print(f"⚠️ 표준용어집 초기화 실패: {e}")
            self.terms_manager = None

        # 2. 문서 처리기 (분리된 데이터 경로 사용)
        self.document_processor = DocumentProcessor(
            data_path=str(self.data_path),
            terms_manager=self.terms_manager
        )
        print("✅ 문서 처리기 초기화 완료")

        # 3. 검색 엔진
        self.search_engine = SearchEngine()
        self.search_engine.set_terms_manager(self.terms_manager)
        print("✅ 검색 엔진 초기화 완료")

        # 4. 캐시 관리자 (분리된 캐시 경로 사용)
        self.cache_manager = CacheManager(cache_path=str(self.cache_path))
        print("✅ 캐시 관리자 초기화 완료")

        # 5. 답변 생성기 (분리된 저장 경로 사용)
        self.answer_generator = AnswerGenerator(
            llm_manager=self.llm_manager,
            save_path=str(self.save_path)
        )
        print("✅ 답변 생성기 초기화 완료 (검색 품질 메트릭 + 관련 검색어 제안 지원)")

        # 시스템 상태
        self.is_initialized = False
        self.data_hash = None

    def initialize_system(self, force_rebuild: bool = False):
        """시스템 초기화"""
        print("🚀 동의보감 RAG 시스템 초기화 중...")

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
                print("🎉 캐시에서 빠르게 로드 완료!")
                return

        # 새로 데이터 처리
        print("📚 새로 데이터를 처리합니다...")

        # 1. 문서 로드 및 청킹
        chunks = self.document_processor.load_documents()

        # 2. 임베딩 생성 및 인덱스 구축
        self.search_engine.setup(chunks)

        # 3. 캐시 저장
        self.cache_manager.save_cache(
            self.data_hash,
            chunks,
            self.search_engine.embeddings,
            self.search_engine.faiss_index
        )

        self.is_initialized = True
        print("✅ 시스템 초기화 완료!")

    def search(self, query: str, k: int = 75):
        """검색 실행"""
        if not self.is_initialized:
            raise ValueError("시스템이 초기화되지 않았습니다.")

        return self.search_engine.search(query, k)

    def generate_answer(self, query: str, search_results):
        """답변 생성"""
        return self.answer_generator.generate_answer(query, search_results)

    def save_results(self, query: str, results, answer: str):
        """결과 저장"""
        self.answer_generator.save_search_results(query, results, answer)

    def _get_k_value_choice(self, recommended_k: int, max_k: int) -> int:
        """K값 선택"""
        print("\n🔧 검색 결과 수를 선택하세요:")
        print(f"1. 권장값 ({recommended_k}개) - 균형잡힌 분석")
        print(f"2. 최대값 ({max_k}개) - 최대한 포괄적 분석")
        print("3. 직접 입력 (50~100)")

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
        """표시 옵션 설정 (관련 검색어 옵션 추가)"""
        print("\n🎨 검색 결과 표시 옵션을 선택하세요:")
        print("1. 카테고리별 + 관련 검색어 (기본, 권장)")
        print("2. 전통적인 리스트 형태")
        print("3. 상세 통계 + 품질 메트릭 + 관련 검색어")
        print("4. 품질 메트릭만 간단히 표시")

        while True:
            choice = input("선택 (1/2/3/4): ").strip()
            if choice == '1':
                return {
                    'show_categories': True,
                    'show_statistics': False,
                    'show_metrics': False,
                    'show_related_queries': True,
                    'traditional_view': False
                }
            elif choice == '2':
                return {
                    'show_categories': False,
                    'show_statistics': False,
                    'show_metrics': False,
                    'show_related_queries': False,
                    'traditional_view': True
                }
            elif choice == '3':
                return {
                    'show_categories': True,
                    'show_statistics': True,
                    'show_metrics': True,
                    'show_related_queries': True,
                    'traditional_view': False
                }
            elif choice == '4':
                return {
                    'show_categories': False,
                    'show_statistics': False,
                    'show_metrics': True,
                    'show_related_queries': False,
                    'traditional_view': False
                }
            else:
                print("1, 2, 3, 또는 4를 입력해주세요.")

    def _process_query(self, query: str, k: int, display_options: dict):
        """쿼리 처리 (관련 검색어 제안 포함) - 수정된 버전"""
        print(f"\n🔍 '{query}' 검색 중... (결과 수: {k}개)")

        # 검색 실행
        search_results = self.search(query, k=k)

        if not search_results:
            print("❌ 관련 내용을 찾을 수 없습니다.")
            print("💡 다른 검색어를 시도해보세요.")
            return

        print(f"📊 {len(search_results)}개의 관련 문서를 찾았습니다.")

        # 답변 생성
        print("🤖 답변 생성 중...")
        answer = self.generate_answer(query, search_results)

        # 결과 표시 방식에 따른 처리
        if display_options.get('show_metrics', False):
            if display_options.get('show_categories', False):
                # 3번 옵션: 카테고리별 + 상세 통계 + 품질 메트릭 + 관련 검색어
                print("\n" + "=" * 50)
                if self.llm_manager and self.llm_manager.is_available():
                    print("🤖 AI 답변 (근거 문헌 주석 포함):")
                    print("-" * 30)
                    print(answer)

                # 자동 저장
                self.save_results(query, search_results, answer)
                print("=" * 50)

                # 카테고리별 검색 결과 표시
                self.answer_generator._display_categorized_results(
                    search_results)

                # 관련 검색어 제안 표시
                if display_options.get('show_related_queries', False):
                    self.answer_generator.display_related_queries(
                        query, search_results)

                # 상세 통계 표시 (옵션)
                if display_options.get('show_statistics', False):
                    self.answer_generator.display_category_statistics(
                        search_results)

                # 품질 메트릭 표시 (고급 분석 포함)
                self.answer_generator.show_search_metrics(
                    query, search_results)

                # 상세 결과 보기 옵션
                show_details_input = input(
                    "\n📋 모든 검색 결과의 전체 내용을 보시겠습니까? (y/n): ").strip().lower()
                if show_details_input in ['y', 'yes', 'ㅇ', '네', '예']:
                    self.answer_generator._display_detailed_results(
                        search_results)

            else:
                # 4번 옵션: 품질 메트릭만 간단히 표시
                print("\n" + "=" * 50)
                if self.llm_manager and self.llm_manager.is_available():
                    print("🤖 AI 답변:")
                    print("-" * 30)
                    print(answer)

                # 자동 저장
                self.save_results(query, search_results, answer)

                # 품질 메트릭만 표시 (고급 분석 포함)
                self.answer_generator.show_search_metrics(
                    query, search_results)

                # 상세 결과 보기 옵션
                show_details_input = input(
                    "\n📋 검색 결과의 상세 내용을 보시겠습니까? (y/n): ").strip().lower()
                if show_details_input in ['y', 'yes', 'ㅇ', '네', '예']:
                    # 카테고리별 표시 후 상세 결과
                    self.answer_generator._display_categorized_results(
                        search_results)

                    detail_input = input(
                        "\n📋 모든 검색 결과의 전체 내용을 보시겠습니까? (y/n): ").strip().lower()
                    if detail_input in ['y', 'yes', 'ㅇ', '네', '예']:
                        self.answer_generator._display_detailed_results(
                            search_results)

        elif display_options.get('show_categories', False):
            # 1번 옵션: 카테고리별 표시 + 관련 검색어 (메트릭 없음)
            show_details = self.answer_generator.display_search_results(
                query, search_results, answer,
                show_related_queries=display_options.get('show_related_queries', True))

            # 상세 통계 표시 (옵션)
            if display_options.get('show_statistics', False):
                self.answer_generator.display_category_statistics(
                    search_results)

        elif display_options.get('traditional_view', False):
            # 2번 옵션: 전통적인 리스트 표시
            show_details = self._display_traditional_results(
                query, search_results, answer)
            # 전통 방식에서도 자동 저장
            self.save_results(query, search_results, answer)

        # 🆕 관련 검색어 선택 처리 추가 (맨 끝에 추가)
        if display_options.get('show_related_queries', False):
            # 관련 검색어 제안이 표시되었으면 사용자 선택 받기
            categorized_suggestions = self.answer_generator.suggest_related_queries(
                query, search_results)
            if categorized_suggestions:
                selected_query = self.answer_generator.get_user_choice_for_suggestions(
                    categorized_suggestions)
                if selected_query and selected_query != query:
                    print(f"\n🔄 '{selected_query}'로 새로운 검색을 시작합니다...")
                    # 재귀 호출로 새로운 검색 실행
                    self._process_query(selected_query, k, display_options)
                    return  # 여기서 리턴해서 아래 코드 실행 방지

    def _handle_related_query_selection(self, query: str, search_results: List[Dict], k: int, display_options: dict) -> Optional[str]:
        """관련 검색어 선택 처리"""
        categorized_suggestions = self.answer_generator.suggest_related_queries(
            query, search_results)

        if not categorized_suggestions:
            return None

        selected_query = self.answer_generator.get_user_choice_for_suggestions(
            categorized_suggestions)

        if selected_query and selected_query != query:
            print(f"\n🔄 '{selected_query}'로 새로운 검색을 시작합니다...")
            self._process_query(selected_query, k, display_options)
            return selected_query

        return None

    def _handle_special_commands(self, query: str) -> bool:
        """특수 명령어 처리 (관련 검색어 명령어 추가)"""
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

        return False

    def _show_related_queries_help(self):
        """관련 검색어 제안 도움말"""
        print("\n🔍 관련 검색어 제안 기능 설명")
        print("=" * 40)

        print("\n💡 기능 개요:")
        print("   • 현재 검색 결과를 분석하여 관련된 검색어를 자동 제안")
        print("   • 카테고리별로 체계적으로 분류된 제안사항 제공")
        print("   • 번호 선택 또는 직접 입력으로 즉시 새로운 검색 가능")

        print("\n🏷️ 제안 카테고리:")
        print("   • 🔥 핵심 처방: 검색 결과에 포함된 주요 처방들")
        print("   • 🩺 관련 병증: 관련된 증상이나 병증들")
        print("   • 💊 주요 약재: 검색 결과에 자주 언급되는 약재들")
        print("   • 📚 관련 개념: 관련된 중의학 이론이나 개념들")
        print("   • 🎯 맞춤 제안: 현재 검색어와 맥락을 고려한 추천")

        print("\n🔄 사용 방법:")
        print("   1. 검색 후 관련 검색어 제안 목록 확인")
        print("   2. 번호를 입력하여 바로 해당 검색어로 검색")
        print("   3. 새로운 검색어를 직접 입력하여 검색")
        print("   4. Enter로 관련 검색어 건너뛰기")

        print("\n✨ 활용 팁:")
        print("   • 처방 검색 후 → 구성 약재나 관련 병증 탐색")
        print("   • 병증 검색 후 → 치료 처방이나 감별진단 확인")
        print("   • 약재 검색 후 → 관련 처방이나 효능 비교")
        print("   • 이론 검색 후 → 임상 응용이나 구체적 사례 탐색")

    def _show_help(self):
        """도움말 표시 (관련 검색어 기능 추가)"""
        print("\n📚 동의보감 RAG 시스템 도움말")
        print("=" * 50)
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
        print("   - quit, exit, 종료: 시스템 종료")
        print()
        print("🆕 새로운 기능:")
        print("   - 🔍 관련 검색어 제안: 검색 결과 기반 스마트 추천")
        print("   - 🏷️ 카테고리별 분류: 처방, 병증, 약재, 이론별 정리")
        print("   - 📊 검색 품질 메트릭: 실시간 검색 성능 분석")
        print("   - 🎯 맞춤형 제안: 컨텍스트 기반 지능형 추천")
        print()
        print("💡 팁:")
        print("   - 한자와 한글 모두 검색 가능")
        print("   - 처방명, 약물명, 병증명 등으로 검색")
        print("   - 관련 검색어로 연관 정보 탐색")
        print("   - 검색 품질 메트릭으로 결과 품질 확인")

    def _show_system_stats(self):
        """시스템 통계 표시"""
        info = self.get_system_info()
        print("\n📊 시스템 통계")
        print("=" * 40)
        print(f"📚 총 청크 수: {info.get('chunks_count', 'N/A'):,}개")
        print(f"🔢 임베딩 차원: {info.get('embeddings_shape', 'N/A')}")
        print(
            f"💾 캐시 상태: {'활성' if info.get('cache_info', {}).get('cache_complete', False) else '비활성'}")
        print(f"📂 데이터 경로: {info['data_path']}")
        print(f"💾 캐시 경로: {info['cache_path']}")
        print(f"💾 결과 저장 경로: {info['save_path']}")
        print(f"🔗 용어집 연결: {'✅' if info['terms_manager_available'] else '❌'}")

        # LLM 정보
        model_info = self.llm_manager.get_model_info()
        print(f"🤖 AI 모델: {model_info['display_name']}")
        print(f"🎯 최적 K값: {model_info['optimal_k']}")

        # 관련 검색어 기능 상태
        print(f"🔍 관련 검색어 기능: ✅ 활성")
        print(f"📊 검색 품질 메트릭: ✅ 활성")

    def _show_config_menu(self):
        """설정 메뉴 표시"""
        print("\n⚙️ 설정 메뉴")
        print("=" * 30)
        print("1. 캐시 정보 확인")
        print("2. 캐시 삭제")
        print("3. 표준용어집 정보")
        print("4. 관련 검색어 기능 테스트")
        print("5. 돌아가기")

        while True:
            choice = input("선택 (1/2/3/4/5): ").strip()
            if choice == '1':
                cache_info = self.cache_manager.get_cache_info()
                print(f"\n💾 캐시 정보:")
                print(
                    f"   상태: {'완전' if cache_info['cache_complete'] else '불완전'}")
                if 'chunks_count' in cache_info:
                    print(f"   청크 수: {cache_info['chunks_count']:,}개")
                    print(f"   생성 시간: {cache_info.get('timestamp', 'N/A')}")
                break
            elif choice == '2':
                confirm = input("정말로 캐시를 삭제하시겠습니까? (y/n): ").lower()
                if confirm in ['y', 'yes', 'ㅇ']:
                    self.cache_manager.clear_cache()
                    print("🗑️ 캐시가 삭제되었습니다. 다음 실행 시 새로 생성됩니다.")
                break
            elif choice == '3':
                if self.terms_manager:
                    stats = self.terms_manager.get_statistics()
                    print(f"\n📚 표준용어집 정보:")
                    print(f"   총 용어 수: {stats.get('total_terms', 'N/A'):,}개")
                    print(
                        f"   검색 인덱스: {stats.get('search_index_size', 'N/A'):,}개")
                else:
                    print("❌ 표준용어집이 연결되지 않았습니다.")
                break
            elif choice == '4':
                print("\n🧪 관련 검색어 기능 테스트")
                test_query = input("테스트용 검색어를 입력하세요 (예: 血虛): ").strip()
                if test_query and self.is_initialized:
                    print("🔄 테스트 검색 실행 중...")
                    test_results = self.search(test_query, k=20)
                    if test_results:
                        print("✅ 검색 완료! 관련 검색어 제안:")
                        self.answer_generator.display_related_queries(
                            test_query, test_results)
                    else:
                        print("❌ 검색 결과가 없습니다.")
                break
            elif choice == '5':
                break
            else:
                print("1, 2, 3, 4, 또는 5를 입력해주세요.")

    def _show_metrics_help(self):
        """검색 품질 메트릭 도움말"""
        print("\n📊 검색 품질 메트릭 설명")
        print("=" * 40)

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

        print("\n🎯 품질 등급:")
        print("   • S (최우수): 매우 포괄적이고 정확한 검색")
        print("   • A (우수): 균형잡힌 좋은 검색 결과")
        print("   • B (양호): 적절하나 일부 개선 여지")
        print("   • C (보통): 기본적 결과, 개선 필요")
        print("   • D (미흡): 검색 전략 재고 필요")

        print("\n💡 활용 팁:")
        print("   • 등급이 낮으면 검색어를 더 구체적으로 입력")
        print("   • 출처 다양성이 낮으면 더 일반적인 용어 사용")
        print("   • 직접 매칭률이 낮으면 정확한 한자 표기 확인")

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
        """대화형 인터페이스 (관련 검색어 기능 포함)"""
        if not self.is_initialized:
            print("❌ 시스템이 초기화되지 않았습니다.")
            return

        llm_manager = self.llm_manager
        model_info = llm_manager.get_model_info()

        # OpenAI GPT-4o-mini에 최적화된 K값 설정
        recommended_k = 75  # GPT-4o-mini 표준 권장값
        max_k = 100        # 최대값

        print("\n" + "=" * 60)
        print("🏥 동의보감 RAG 시스템 v2.2에 오신 것을 환영합니다!")
        print(f"🤖 사용 중인 모델: {model_info['display_name']}")
        print(f"📊 권장 검색 결과 수: {recommended_k}개 (최대 {max_k}개)")
        print("🆕 새로운 기능:")
        print("   • 카테고리별 검색 결과 그룹핑")
        print("   • 🔥 스마트 관련 검색어 제안 (NEW!)")
        print("   • 검색 품질 메트릭 및 개선 제안")
        print("   • 실시간 검색 성능 분석")
        print("💡 중의학 관련 질문을 한국어로 입력하세요.")
        print("🆘 도움말: 'help' 또는 '도움말' 입력")
        print("🔍 관련 검색어 설명: 'related' 또는 '관련' 입력")
        print("🚪 종료하려면 'quit', 'exit', '종료'를 입력하세요.")

        # K값 설정
        selected_k = self._get_k_value_choice(recommended_k, max_k)
        print(f"✅ 검색 결과 수: {selected_k}개로 설정되었습니다.")

        # 표시 옵션 설정 (관련 검색어 옵션 포함)
        display_options = self._get_display_options()
        print("=" * 60 + "\n")

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

                # 검색 및 답변 생성 (관련 검색어 포함)
                self._process_query(query, selected_k, display_options)

                # 🗑️ 기존의 중복된 관련 검색어 처리 부분 완전 삭제
                # (모든 관련 검색어 처리는 이제 _process_query에서 담당)

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
        # 기존 메서드 내용 유지하되, 경로 정보 업데이트
        info = {
            'initialized': self.is_initialized,
            'data_path': str(self.data_path),
            'cache_path': str(self.cache_path),
            'save_path': str(self.save_path),
            'data_hash': self.data_hash,
            'terms_manager_available': self.terms_manager is not None,
            'cache_info': self.cache_manager.get_cache_info(),
            'related_queries_enabled': True,
            'search_metrics_enabled': True
        }

        if self.is_initialized and hasattr(self.search_engine, 'chunks'):
            info['chunks_count'] = len(self.search_engine.chunks)
            if hasattr(self.search_engine, 'embeddings'):
                info['embeddings_shape'] = self.search_engine.embeddings.shape
            if hasattr(self.search_engine, 'faiss_index'):
                info['faiss_index_count'] = self.search_engine.faiss_index.ntotal

        return info


def main():
    """메인 함수 (관련 검색어 통합 최종 버전)"""
    try:
        print("🏥 동의보감 RAG 시스템 v2.2 (스마트 관련 검색어 제안 통합)")
        print("=" * 60)
        print("🆕 주요 기능:")
        print("   • OpenAI GPT-4o-mini 기반 고품질 답변")
        print("   • 🔥 스마트 관련 검색어 제안 (NEW!)")
        print("   • 카테고리별 검색 결과 그룹핑")
        print("   • 실시간 검색 품질 메트릭 분석")
        print("   • 자동 검색 개선 제안")
        print("   • 근거 문헌 주석 시스템")
        print("   • 🎯 연속 탐색형 검색 인터페이스")

        print("\n🔍 관련 검색어 제안 기능:")
        print("   • 검색 결과 기반 지능형 추천")
        print("   • 처방 → 약재 → 병증 연결 탐색")
        print("   • 카테고리별 체계적 분류")
        print("   • 번호 선택으로 즉시 검색")

        # 시작 옵션 선택
        print("\n🔧 시작 옵션을 선택하세요:")
        print("1. 캐시에서 로드 (빠름, 권장)")
        print("2. 강제 재구축 (느림, 데이터 변경시)")

        while True:
            choice = input("선택 (1/2): ").strip()
            if choice == '1':
                force_rebuild = False
                break
            elif choice == '2':
                force_rebuild = True
                break
            else:
                print("1 또는 2를 입력해주세요.")

        # RAG 시스템 초기화 (OpenAI + 관련 검색어 기능 포함)
        rag_system = DonguiRAGSystem()
        rag_system.initialize_system(force_rebuild=force_rebuild)

        # 대화형 인터페이스 시작
        rag_system.interactive_chat()

    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("💡 해결 방법:")
        print("   1. OpenAI API 키 확인 (.env 파일 또는 환경변수)")
        print("   2. 필수 모듈 설치 확인 (requirements.txt)")
        print("   3. 데이터 경로 확인 (/Users/radi/Projects/langchain/DATA/DYBG)")
        print("   4. 표준용어집 파일 확인 (hmedicalterms.json)")
        sys.exit(1)


if __name__ == "__main__":
    main()
