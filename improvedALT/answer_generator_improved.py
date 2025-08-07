#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
답변 생성 모듈 - answer_generator.py (개선된 버전 - 하드코딩 제거)
LLM을 이용한 답변 생성과 결과 저장을 담당
표준한의학용어집 기반으로 동적 패턴 생성 및 관련 검색어 제안
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")


class AnswerGenerator:
    def __init__(self, llm_manager=None, save_path: str = "/Users/radi/Projects/langchainDATA/Results/DYBGsearch", terms_manager=None):
        """답변 생성기 초기화 (개선된 버전)"""
        self.llm_manager = llm_manager
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # 표준용어집 관리자 추가
        self.terms_manager = terms_manager

        # 관련 검색어 추출을 위한 패턴들
        self.prescription_patterns = [
            r'([一-龯]{2,6}[湯散丸膏])',  # 처방명 패턴
            r'([一-龯]{3,8})'           # 일반 처방명
        ]

        self.symptom_patterns = [
            r'([一-龯]{2,4}[證病症])',    # 증상/병증 패턴
            r'([一-龯]{1,3}[虛實])',     # 허실 패턴
            r'([一-龯]{2,4}[痛])',       # 통증 패턴
            r'([一-龯]{2,4}[熱寒])',     # 한열 패턴
        ]

        self.herb_patterns = [
            r'([一-龯]{2,4}[參芎歸芍地黃茯苓芪朮])',  # 약재명 패턴
            r'([一-龯]{2,4})',                        # 일반 약재명
        ]

    def set_terms_manager(self, terms_manager):
        """표준용어집 관리자 설정"""
        self.terms_manager = terms_manager
        # 캐시 무효화
        self._invalidate_cache()

    def _invalidate_cache(self):
        """패턴 캐시 무효화"""
        self._prescription_patterns_cache = None
        self._symptom_patterns_cache = None
        self._herb_patterns_cache = None
        self._cache_timestamp = None

    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if self._cache_timestamp is None:
            return False

        from datetime import timedelta
        now = datetime.now()
        cache_age = now - self._cache_timestamp
        return cache_age < timedelta(hours=self._cache_validity_hours)

    def _get_prescription_patterns(self) -> List[str]:
        """처방 패턴 동적 생성 (용어집 기반)"""
        if self._prescription_patterns_cache and self._is_cache_valid():
            return self._prescription_patterns_cache

        patterns = []

        try:
            if self.terms_manager:
                # 용어집에서 처방 분류 추출
                prescriptions = self.terms_manager.search_by_category('처방')

                # 처방 종류별 패턴 분석
                suffixes = set()
                for prescription in prescriptions:
                    hanja_name = prescription.get('용어명_한자', '')
                    if hanja_name:
                        # 처방 접미사 추출
                        if hanja_name.endswith(('湯', '散', '丸', '膏', '飮', '丹', '露')):
                            suffixes.add(hanja_name[-1])

                # 동적 패턴 생성
                for suffix in suffixes:
                    patterns.append(f'([一-龯]{{2,8}}{suffix})')

                # 일반적인 처방명 패턴도 추가
                patterns.append(r'([一-龯]{3,8}方)')
                patterns.append(r'([一-龯]{3,8}劑)')

            # 폴백: 기본 패턴
            if not patterns:
                patterns = [
                    r'([一-龯]{2,6}[湯散丸膏])',
                    r'([一-龯]{3,8})'
                ]

        except Exception as e:
            print(f"⚠️ 처방 패턴 생성 실패: {e}")
            # 폴백 패턴
            patterns = [
                r'([一-龯]{2,6}[湯散丸膏])',
                r'([一-龯]{3,8})'
            ]

        # 캐시 저장
        self._prescription_patterns_cache = patterns
        self._cache_timestamp = datetime.now()

        return patterns

    def _get_symptom_patterns(self) -> List[str]:
        """증상/병증 패턴 동적 생성 (용어집 기반)"""
        if self._symptom_patterns_cache and self._is_cache_valid():
            return self._symptom_patterns_cache

        patterns = []

        try:
            if self.terms_manager:
                # 용어집에서 병증 분류 추출
                symptoms = self.terms_manager.search_by_category('병증')

                # 병증 접미사 분석
                suffixes = set()
                for symptom in symptoms:
                    hanja_name = symptom.get('용어명_한자', '')
                    if hanja_name:
                        # 병증 접미사 추출
                        if hanja_name.endswith(('證', '病', '症', '痛', '虛', '實', '熱', '寒')):
                            suffixes.add(hanja_name[-1])

                # 동적 패턴 생성
                for suffix in suffixes:
                    if suffix in ['虛', '實']:
                        patterns.append(f'([一-龯]{{1,4}}{suffix})')
                    elif suffix in ['痛']:
                        patterns.append(f'([一-龯]{{2,4}}{suffix})')
                    else:
                        patterns.append(f'([一-龯]{{2,5}}{suffix})')

            # 폴백: 기본 패턴
            if not patterns:
                patterns = [
                    r'([一-龯]{2,4}[證病症])',
                    r'([一-龯]{1,3}[虛實])',
                    r'([一-龯]{2,4}[痛])',
                    r'([一-龯]{2,4}[熱寒])'
                ]

        except Exception as e:
            print(f"⚠️ 증상 패턴 생성 실패: {e}")
            patterns = [
                r'([一-龯]{2,4}[證病症])',
                r'([一-龯]{1,3}[虛實])',
                r'([一-龯]{2,4}[痛])',
                r'([一-龯]{2,4}[熱寒])'
            ]

        # 캐시 저장
        self._symptom_patterns_cache = patterns
        if not self._cache_timestamp:
            self._cache_timestamp = datetime.now()

        return patterns

    def _get_herb_patterns(self) -> List[str]:
        """약재 패턴 동적 생성 (용어집 기반)"""
        if self._herb_patterns_cache and self._is_cache_valid():
            return self._herb_patterns_cache

        patterns = []

        try:
            if self.terms_manager:
                # 용어집에서 약물 분류 추출
                herbs = self.terms_manager.search_by_category('약물')

                # 약재명 특성 분석
                common_chars = set()
                for herb in herbs[:100]:  # 상위 100개만 분석
                    hanja_name = herb.get('용어명_한자', '')
                    if hanja_name and len(hanja_name) >= 2:
                        # 마지막 글자 수집 (약재 특성)
                        common_chars.add(hanja_name[-1])

                # 빈도 높은 약재 특성 글자들로 패턴 생성
                herb_chars = ['參', '芎', '歸', '芍', '地', '黃',
                              '茯', '苓', '芪', '朮', '草', '皮', '仁', '子']

                for char in herb_chars:
                    if char in common_chars:
                        patterns.append(f'([一-龯]{{2,4}}{char})')

                # 일반적인 약재 패턴
                patterns.append(r'([一-龯]{2,4})')

            # 폴백: 기본 패턴
            if not patterns:
                patterns = [
                    r'([一-龯]{2,4}[參芎歸芍地黃茯苓芪朮])',
                    r'([一-龯]{2,4})'
                ]

        except Exception as e:
            print(f"⚠️ 약재 패턴 생성 실패: {e}")
            patterns = [
                r'([一-龯]{2,4}[參芎歸芍地黃茯苓芪朮])',
                r'([一-龯]{2,4})'
            ]

        # 캐시 저장
        self._herb_patterns_cache = patterns
        if not self._cache_timestamp:
            self._cache_timestamp = datetime.now()

        return patterns

    def _get_major_herbs_from_terms(self) -> List[str]:
        """용어집에서 주요 약재 추출"""
        try:
            if not self.terms_manager:
                return self._get_fallback_herbs()

            herbs = self.terms_manager.search_by_category('약물')
            major_herbs = []

            # 상위 50개 약재 추출
            for herb in herbs[:50]:
                hanja_name = herb.get('용어명_한자', '')
                if hanja_name:
                    major_herbs.append(hanja_name)

            return major_herbs if major_herbs else self._get_fallback_herbs()

        except Exception as e:
            print(f"⚠️ 용어집에서 약재 추출 실패: {e}")
            return self._get_fallback_herbs()

    def _get_key_concepts_from_terms(self) -> List[str]:
        """용어집에서 핵심 개념 추출"""
        try:
            if not self.terms_manager:
                return self._get_fallback_concepts()

            concepts = []

            # 생리, 이론 분류에서 추출
            for category in ['생리', '이론', '변증']:
                terms = self.terms_manager.search_by_category(category)
                for term in terms[:20]:  # 각 카테고리당 20개
                    hanja_name = term.get('용어명_한자', '')
                    if hanja_name:
                        concepts.append(hanja_name)

            return concepts if concepts else self._get_fallback_concepts()

        except Exception as e:
            print(f"⚠️ 용어집에서 개념 추출 실패: {e}")
            return self._get_fallback_concepts()

    def _get_symptom_keywords_from_terms(self) -> List[str]:
        """용어집에서 증상 키워드 추출"""
        try:
            if not self.terms_manager:
                return self._get_fallback_symptoms()

            symptoms = self.terms_manager.search_by_category('병증')
            symptom_keywords = []

            for symptom in symptoms[:30]:  # 상위 30개
                hanja_name = symptom.get('용어명_한자', '')
                if hanja_name:
                    symptom_keywords.append(hanja_name)

            return symptom_keywords if symptom_keywords else self._get_fallback_symptoms()

        except Exception as e:
            print(f"⚠️ 용어집에서 증상 추출 실패: {e}")
            return self._get_fallback_symptoms()

    def _get_fallback_herbs(self) -> List[str]:
        """폴백용 기본 약재 리스트"""
        return [
            '人參', '當歸', '川芎', '白芍', '熟地黃', '生地黃', '黃芪', '白朮',
            '茯苓', '甘草', '陳皮', '半夏', '枳實', '厚朴', '桔梗', '杏仁',
            '麥門冬', '五味子', '山藥', '茯神', '遠志', '石菖蒲', '朱砂', '龍骨',
            '牡蠣', '酸棗仁', '柏子仁', '阿膠', '地骨皮', '知母', '黃柏', '山茱萸'
        ]

    def _get_fallback_concepts(self) -> List[str]:
        """폴백용 기본 개념 리스트"""
        return [
            '陰陽', '五行', '臟腑', '氣血', '經絡', '精氣神', '君臣佐使',
            '四象', '八綱', '六經', '營衛', '三焦', '命門', '元氣', '真陰',
            '火神', '溫補', '滋陰', '理氣', '活血', '化痰', '祛濕', '清熱'
        ]

    def _get_fallback_symptoms(self) -> List[str]:
        """폴백용 기본 증상 리스트"""
        return [
            '驚悸', '健忘', '眩暈', '失眠', '虛勞', '血虛', '氣虛', '陰虛',
            '陽虛', '脾胃虛', '心悸', '不寐', '頭痛', '腹痛', '胸痛'
        ]

    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """답변 생성 (근거 문헌 주석 강화)"""
        if not self.llm_manager or not self.llm_manager.is_available():
            return "LLM이 연결되지 않아 답변을 생성할 수 없습니다. 검색 결과를 확인해주세요."

        # 컨텍스트 구성 (문서 번호와 함께)
        context_parts = []
        for i, result in enumerate(search_results):
            # 문서 번호를 명확히 포함
            context_parts.append(
                f"[문서 {i + 1}]\n출처: {result['metadata'].get('source_file', 'unknown')}\n내용: {result['content']}")

        # LLM 관리자를 통한 컨텍스트 최적화
        optimized_context_parts = self.llm_manager.optimize_context_for_model(
            context_parts)
        context = '\n\n'.join(optimized_context_parts)

        # 강화된 시스템 프롬프트
        system_prompt = self._get_enhanced_system_prompt()

        # 강화된 사용자 프롬프트
        user_prompt = self._get_enhanced_user_prompt(
            query, context, len(search_results))

        # 메시지 구성 및 응답 생성
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        except ImportError:
            # 폴백: 기본 메시지 형식
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        return self.llm_manager.generate_response(messages)

    def _extract_prescriptions_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 처방명 추출 (개선된 버전)"""
        prescriptions = []
        prescription_counts = Counter()

        for result in results:
            # 메타데이터에서 처방명 직접 추출
            if result['metadata'].get('prescription_name'):
                prescription_counts[result['metadata']
                                    ['prescription_name']] += 3

            # 내용에서 처방명 패턴 매칭 (동적 패턴 사용)
            content = result['content']
            patterns = self._get_prescription_patterns()

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if len(match) >= 3:  # 최소 3글자 이상
                            prescription_counts[match] += 1
                except re.error:
                    continue

        # 빈도순으로 정렬하여 상위 항목 반환
        return [name for name, count in prescription_counts.most_common(6) if count >= 2]

    def _extract_symptoms_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 증상/병증 추출 (개선된 버전)"""
        symptoms = []
        symptom_counts = Counter()

        for result in results:
            content = result['content']

            # 증상/병증 패턴 매칭 (동적 패턴 사용)
            patterns = self._get_symptom_patterns()

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if len(match) >= 2:
                            symptom_counts[match] += 1
                except re.error:
                    continue

            # 용어집 기반 증상 키워드 매칭
            symptom_keywords = self._get_symptom_keywords_from_terms()
            for keyword in symptom_keywords:
                if keyword in content:
                    symptom_counts[keyword] += 2

        return [symptom for symptom, count in symptom_counts.most_common(8) if count >= 2]

    def _extract_herbs_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 약재명 추출 (개선된 버전)"""
        herbs = []
        herb_counts = Counter()

        # 용어집에서 주요 약재 리스트 가져오기
        major_herbs = self._get_major_herbs_from_terms()

        for result in results:
            content = result['content']

            # 주요 약재 검색
            for herb in major_herbs:
                if herb in content:
                    # 처방 구성에 나오면 가중치 부여
                    if any(keyword in content for keyword in ['右', '右爲末', '右剉', '右爲']):
                        herb_counts[herb] += 3
                    else:
                        herb_counts[herb] += 1

        return [herb for herb, count in herb_counts.most_common(8) if count >= 2]

    def _extract_concepts_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 이론/개념 추출 (개선된 버전)"""
        concepts = []
        concept_counts = Counter()

        # 용어집에서 핵심 개념들 가져오기
        key_concepts = self._get_key_concepts_from_terms()

        for result in results:
            content = result['content']
            bb = result['metadata'].get('BB', '')
            cc = result['metadata'].get('CC', '')

            # 대분류/중분류에서 개념 추출
            if bb and bb not in ['', 'unknown']:
                concept_counts[bb] += 2
            if cc and cc not in ['', 'unknown']:
                concept_counts[cc] += 1

            # 핵심 개념 매칭
            for concept in key_concepts:
                if concept in content:
                    concept_counts[concept] += 1

        return [concept for concept, count in concept_counts.most_common(5) if count >= 2]

    def _get_contextual_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """컨텍스트 기반 맞춤 제안 (용어집 연동 개선)"""
        suggestions = []

        # 1. 용어집 기반 쿼리 분석
        try:
            if self.terms_manager:
                # 쿼리의 용어집 정보 조회
                query_info = self.terms_manager.get_term_info(query)
                if query_info:
                    category = query_info.get('분류', '')

                    # 분류별 맞춤 제안
                    if category == '병증':
                        suggestions.extend(['치료방법', '감별진단', '병리기전'])
                    elif category == '처방':
                        suggestions.extend(['가감방', '배합금기', '용법용량'])
                    elif category == '약물':
                        suggestions.extend(['약성', '귀경', '효능주치', '배합'])
                    elif category in ['생리', '이론']:
                        suggestions.extend(['임상응용', '관련이론', '실용방법'])

                # 관련 용어 추가
                related_terms = self.terms_manager.get_related_terms(query)
                suggestions.extend(related_terms[:3])

        except Exception as e:
            print(f"⚠️ 용어집 기반 제안 실패: {e}")

        # 2. 쿼리 유형 분석 기반 제안 (폴백)
        if '虛' in query:
            suggestions.extend(['補益', '溫陽', '滋陰', '益氣'])
        elif '湯' in query:
            suggestions.extend(['加減方', '배합금기', '용법용량'])
        elif '病' in query or '證' in query:
            suggestions.extend(['治療方법', '감별진단', '병리기전'])
        elif any(herb in query for herb in ['人參', '當歸', '川芎']):
            suggestions.extend(['약성', '귀경', '효능주치', '배합'])

        # 3. 검색 결과 메타데이터 기반 제안
        source_files = set()
        bb_categories = set()

        for result in results:
            source_file = result['metadata'].get('source_file', '')
            bb = result['metadata'].get('BB', '')

            if source_file:
                # 파일명에서 관련 주제 추출
                if '내경편' in source_file:
                    suggestions.append('정신요법')
                elif '외형편' in source_file:
                    suggestions.append('침구치료')
                elif '잡병편' in source_file:
                    suggestions.append('임상응용')
                elif '탕액편' in source_file:
                    suggestions.append('본초학')

            if bb and bb not in bb_categories:
                bb_categories.add(bb)
                # BB 카테고리 기반 관련 주제 제안
                if bb == '血':
                    suggestions.extend(['補血', '活血', '止血'])
                elif bb == '氣':
                    suggestions.extend(['補氣', '理氣', '降氣'])
                elif bb == '精':
                    suggestions.extend(['補腎', '固精', '滋陰'])

        # 4. 빈도 기반 필터링 및 중복 제거
        suggestion_counts = Counter(suggestions)
        final_suggestions = []

        for suggestion, count in suggestion_counts.most_common(5):
            if suggestion and suggestion not in final_suggestions:
                final_suggestions.append(suggestion)

        return final_suggestions

    def _is_too_similar_to_query(self, query: str, suggestion: str) -> bool:
        """제안어가 검색어와 너무 유사한지 확인"""
        if query in suggestion or suggestion in query:
            return True

        # 글자 겹침 비율 확인
        common_chars = set(query) & set(suggestion)
        similarity_ratio = len(common_chars) / \
            max(len(set(query)), len(set(suggestion)))

        return similarity_ratio > 0.8

    def suggest_related_queries(self, query: str, results: List[Dict], max_suggestions: int = 8) -> Dict[str, List[str]]:
        """관련 검색어 제안 기능 (용어집 기반으로 개선)"""
        # 1. 검색 결과에서 처방명 추출 (개선된 함수 사용)
        prescriptions = self._extract_prescriptions_from_results(results)

        # 2. 관련 증상/병증 추출 (개선된 함수 사용)
        symptoms = self._extract_symptoms_from_results(results)

        # 3. 관련 약재 추출 (개선된 함수 사용)
        herbs = self._extract_herbs_from_results(results)

        # 4. 관련 이론/개념 추출 (개선된 함수 사용)
        concepts = self._extract_concepts_from_results(results)

        # 5. 컨텍스트 기반 추천 (개선된 함수 사용)
        contextual_suggestions = self._get_contextual_suggestions(
            query, results)

        # 우선순위별로 추천 목록 구성
        suggestion_categories = [
            ("🔥 핵심 처방", prescriptions[:2]),
            ("🩺 관련 병증", symptoms[:2]),
            ("💊 주요 약재", herbs[:2]),
            ("📚 관련 개념", concepts[:1]),
            ("🎯 맞춤 제안", contextual_suggestions[:1])
        ]

        # 카테고리별로 제안사항 수집
        categorized_suggestions = {}
        all_suggestions = []

        for category, items in suggestion_categories:
            if items:
                # 중복 제거 및 현재 검색어와 다른 것만 선택
                filtered_items = []
                for item in items:
                    if (item not in all_suggestions and
                        item.lower() != query.lower() and
                            not self._is_too_similar_to_query(query, item)):
                        filtered_items.append(item)
                        all_suggestions.append(item)

                if filtered_items:
                    categorized_suggestions[category] = filtered_items

        return categorized_suggestions

    def display_related_queries(self, query: str, results: List[Dict]):
        """관련 검색어 제안 표시"""
        print("\n" + "💡" * 25)
        print("🔍 관련 검색 제안")
        print("=" * 50)

        categorized_suggestions = self.suggest_related_queries(query, results)

        if not categorized_suggestions:
            print("💭 현재 검색 결과를 바탕으로 한 관련 검색어를 찾을 수 없습니다.")
            print("🔄 다른 검색어를 시도해보세요.")
            return

        suggestion_count = 1

        for category, suggestions in categorized_suggestions.items():
            if suggestions:
                print(f"\n{category}:")
                for suggestion in suggestions:
                    print(f"   {suggestion_count}. {suggestion}")
                    suggestion_count += 1

        print(f"\n💡 총 {suggestion_count - 1}개의 관련 검색어를 제안합니다.")
        print("🔄 위 번호를 입력하거나 직접 검색어를 입력하세요.")

    def get_user_choice_for_suggestions(self, categorized_suggestions: Dict[str, List[str]]) -> Optional[str]:
        """사용자의 관련 검색어 선택 처리"""
        if not categorized_suggestions:
            return None

        # 모든 제안사항을 평평한 리스트로 변환
        all_suggestions = []
        for suggestions in categorized_suggestions.values():
            all_suggestions.extend(suggestions)

        if not all_suggestions:
            return None

        while True:
            try:
                choice = input(
                    "\n🤔 선택하세요 (번호 입력 또는 새 검색어 입력, Enter로 건너뛰기): ").strip()

                if not choice:  # Enter로 건너뛰기
                    return None

                # 숫자 입력 처리
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(all_suggestions):
                        selected_query = all_suggestions[choice_num - 1]
                        print(f"✅ '{selected_query}'를 선택했습니다.")
                        return selected_query
                    else:
                        print(f"❌ 1~{len(all_suggestions)} 범위의 번호를 입력해주세요.")
                        continue

                # 직접 입력된 검색어 처리
                else:
                    print(f"✅ '{choice}'로 새로운 검색을 시작합니다.")
                    return choice

            except ValueError:
                print("❌ 올바른 번호를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n🚫 선택을 취소합니다.")
                return None

    def _get_enhanced_system_prompt(self) -> str:
        """강화된 시스템 프롬프트 (근거 문헌 주석 필수화)"""
        return """당신은 동의보감 전문가입니다. 제공된 동의보감 원문을 바탕으로 질문에 정확하고 체계적으로 답변하세요.

## 🚨 필수 규칙 (반드시 준수)
1. **모든 내용 뒤에는 반드시 [출처: 문서X] 형태로 근거 문헌을 표시**
2. **여러 문서에서 확인된 내용은 [출처: 문서X,Y,Z] 형태로 모든 문서 번호 표시**
3. **처방 구성, 용법 등 구체적 정보는 정확한 출처 문서 번호 필수**
4. **출처가 불분명한 내용은 절대 작성 금지**

## 📋 답변 원칙
1. **정확성 우선**: 제공된 원문에만 근거하여 답변
2. **체계적 구성**: 논리적이고 이해하기 쉬운 구조
3. **원문 인용**: 중요한 내용은 한자 원문 그대로 인용
4. **실용성 강조**: 임상적으로 활용 가능한 정보 우선 제시
5. **포괄적 분석**: 제공된 다수의 자료를 종합적으로 활용
6. **근거 문헌 표시**: 모든 내용에 대해 근거가 되는 원문 출처를 주석으로 명시

## 🏗️ 답변 구조 (질문 유형에 따라 선택적 적용)

### 📚 이론/개념 질문인 경우:
**1. 정의와 개념** - 동의보감에서의 정의, 관련 이론적 배경 [출처: 문서X,Y]
**2. 병리기전** - 발생 원인과 기전, 관련 장부와 경락 [출처: 문서Z]
**3. 임상 의의** - 진단상 중요성, 다른 개념과의 관계 [출처: 문서A,B]

### 💊 처방 질문인 경우:
**1. 처방 개요** - 처방명과 출전, 주치증과 적응증 [출처: 문서X]
**2. 구성과 용법** - 구성 약물과 분량, 복용법과 주의사항 [출처: 문서Y,Z]
**3. 임상 응용** - 가감법과 변증 요점, 관련 처방들과의 비교 [출처: 문서A]

### 🩺 병증 질문인 경우:
**1. 병증 정의** - 병명과 특징, 분류와 유형 [출처: 문서X,Y]
**2. 증상과 진단** - 주요 증상과 맥상, 감별진단 요점 [출처: 문서Z]
**3. 치료 방법** - 주요 치료 처방, 치료 원칙과 예후 [출처: 문서A,B]

### 🌿 약물 질문인 경우:
**1. 약성과 귀경** - 성미와 독성, 귀경과 작용 부위 [출처: 문서X]
**2. 효능과 주치** - 주요 효능, 치료 가능한 병증 [출처: 문서Y,Z]
**3. 용법과 배합** - 용량과 복용법, 배합 금기와 주의사항 [출처: 문서A]

## 📚 근거 문헌 표시 규칙
- 각 문장이나 단락 끝에 반드시 [출처: 문서X] 형태로 표시
- 여러 문서에서 확인된 내용: [출처: 문서1,3,7]
- 처방명, 약재명, 용법 등: 반드시 정확한 출처 명시
- 예시: "陰虛는 음의 부족을 의미합니다 [출처: 문서1,3]. 주요 증상으로는 발열과 가슴 답답함이 있습니다 [출처: 문서2,5]."

## ✅ 답변 지침
- **종합적 분석**: 많은 검색 결과를 바탕으로 완전한 그림 제시
- **핵심부터 제시**: 가장 중요한 정보를 먼저 설명
- **원문 인용**: 중요한 부분은 "○○曰, ..." 형태로 한자 인용
- **명확한 한자 표기**: 전문 용어는 한자와 한글 병기
- **문서 번호 정확성**: 제공된 문서 번호와 정확히 일치

## ⚠️ 주의사항
- **모든 문장에 출처 표시 필수** - 출처 없는 내용은 절대 작성 금지
- 제공된 자료에 없는 내용 추가 금지
- 현대 의학적 해석이나 개인적 견해 첨가 금지
- 불확실한 경우 솔직히 인정하되 출처는 반드시 표시

이러한 지침에 따라 동의보감의 전문성을 유지하면서도 모든 내용의 출처를 명확히 표시한 답변을 제공하세요."""

    def _get_enhanced_user_prompt(self, query: str, context: str, result_count: int) -> str:
        """강화된 사용자 프롬프트"""
        return f"""다음 동의보감 원문을 참고하여 질문에 답변해주세요:

    === 검색된 동의보감 원문 ({result_count}개 문서) ===
    {context}

    === 질문 ===
    {query}

    === 답변 요청 ===
    🚨 **필수 요구사항**:
    - **모든 내용 뒤에 반드시 [출처: 문서X] 또는 [출처: 문서X,Y,Z] 형태로 근거 문헌을 표시하세요**
    - **출처가 불분명한 내용은 절대 작성하지 마세요**
    - **처방 구성, 증상, 치료법 등 모든 구체적 정보에는 정확한 문서 번호를 표시하세요**

    위의 강화된 답변 지침에 따라 체계적이고 실용적인 답변을 제공하되, 반드시 모든 내용에 대해 근거가 되는 문서 번호를 명시해주세요. 많은 검색 결과를 종합적으로 활용하여 완전하고 균형잡힌 답변을 작성해주세요."""

    def show_search_metrics(self, query: str, results: List[Dict]):
        """검색 품질 메트릭 표시"""
        if not results:
            print("📊 검색 결과가 없어 메트릭을 표시할 수 없습니다.")
            return

        # 기본 메트릭 계산
        metrics = {
            '처방 정보': len([r for r in results if r['metadata'].get('type') == 'prescription']),
            '이론 내용': len([r for r in results if r['metadata'].get('BB')]),
            '출처 다양성': len(set(r['metadata'].get('source_file', 'unknown') for r in results)),
            '평균 관련도': sum(r.get('score', 0) for r in results) / len(results)
        }

        # 고급 메트릭 계산
        advanced_metrics = self._calculate_advanced_metrics(query, results)

        print(f"\n📊 검색 품질 지표 ('{query}' 검색 결과):")
        print("=" * 50)

        # 기본 메트릭 표시
        print("🔍 기본 지표:")
        for metric, value in metrics.items():
            if metric == '평균 관련도':
                print(f"   • {metric}: {value:.3f}")
            else:
                print(f"   • {metric}: {value}개")

        print()

        # 고급 메트릭 표시
        print("📈 고급 분석:")
        for metric, value in advanced_metrics.items():
            if isinstance(value, float):
                print(f"   • {metric}: {value:.3f}")
            elif isinstance(value, list):
                print(f"   • {metric}: {', '.join(map(str, value))}")
            else:
                print(f"   • {metric}: {value}")

        # 품질 등급 평가
        quality_grade = self._evaluate_search_quality(
            metrics, advanced_metrics)
        print(
            f"\n🎯 검색 품질 등급: {quality_grade['grade']} ({quality_grade['description']})")

        # 개선 제안
        suggestions = self._get_improvement_suggestions(
            query, metrics, advanced_metrics)
        if suggestions:
            print(f"\n💡 검색 개선 제안:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")

    def _calculate_advanced_metrics(self, query: str, results: List[Dict]) -> Dict:
        """고급 메트릭 계산"""
        advanced_metrics = {}

        # 1. 내용 타입별 분포
        content_types = defaultdict(int)
        for result in results:
            metadata = result['metadata']
            if metadata.get('type') == 'prescription':
                content_types['처방'] += 1
            elif metadata.get('BB'):
                content_types['이론'] += 1
            elif any(kw in result['content'] for kw in ['證', '病', '症']):
                content_types['병증'] += 1
            elif any(kw in result['content'] for kw in ['味', '性', '歸經']):
                content_types['약물'] += 1
            else:
                content_types['기타'] += 1

        advanced_metrics['내용 타입 다양성'] = len(content_types)

        # 2. 관련도 점수 분포
        scores = [r.get('score', 0) for r in results]
        if scores:
            advanced_metrics['최고 관련도'] = max(scores)
            advanced_metrics['최저 관련도'] = min(scores)
            advanced_metrics['관련도 편차'] = max(scores) - min(scores)

            # 관련도 구간별 분포
            high_quality = len([s for s in scores if s >= 3.0])
            medium_quality = len([s for s in scores if 1.5 <= s < 3.0])
            low_quality = len([s for s in scores if s < 1.5])

            advanced_metrics['고품질 결과 (≥3.0)'] = f"{high_quality}개 ({high_quality / len(results) * 100:.1f}%)"
            advanced_metrics['중품질 결과 (1.5-3.0)'] = f"{medium_quality}개 ({medium_quality / len(results) * 100:.1f}%)"
            advanced_metrics['저품질 결과 (<1.5)'] = f"{low_quality}개 ({low_quality / len(results) * 100:.1f}%)"

        # 3. 출처 파일별 분포
        source_distribution = defaultdict(int)
        for result in results:
            source_file = result['metadata'].get('source_file', 'unknown')
            source_distribution[source_file] += 1

        # 가장 많이 활용된 출처 상위 3개
        top_sources = sorted(source_distribution.items(),
                             key=lambda x: x[1], reverse=True)[:3]
        advanced_metrics['주요 출처'] = [
            f"{source}({count}개)" for source, count in top_sources]

        # 4. 대분류(BB) 다양성
        bb_categories = set()
        for result in results:
            bb = result['metadata'].get('BB')
            if bb:
                bb_categories.add(bb)

        advanced_metrics['대분류 다양성'] = len(bb_categories)
        if bb_categories:
            advanced_metrics['포함된 대분류'] = list(bb_categories)

        # 5. 중분류(CC) 다양성
        cc_categories = set()
        for result in results:
            cc = result['metadata'].get('CC')
            if cc:
                cc_categories.add(cc)

        advanced_metrics['중분류 다양성'] = len(cc_categories)

        # 6. 처방명 다양성 (처방 관련 검색인 경우)
        prescription_names = set()
        for result in results:
            prescription_name = result['metadata'].get('prescription_name')
            if prescription_name:
                prescription_names.add(prescription_name)

        if prescription_names:
            advanced_metrics['처방 다양성'] = len(prescription_names)
            if len(prescription_names) <= 5:
                advanced_metrics['포함된 처방'] = list(prescription_names)

        # 7. 내용 길이 분석
        content_lengths = [len(result['content']) for result in results]
        if content_lengths:
            advanced_metrics['평균 내용 길이'] = sum(
                content_lengths) / len(content_lengths)
            advanced_metrics['내용 길이 범위'] = f"{min(content_lengths)}~{max(content_lengths)}자"

        # 8. 키워드 매칭 품질
        direct_matches = len([r for r in results if query in r['content']])
        advanced_metrics['직접 매칭률'] = f"{direct_matches}개 ({direct_matches / len(results) * 100:.1f}%)"

        return advanced_metrics

    def _evaluate_search_quality(self, basic_metrics: Dict, advanced_metrics: Dict) -> Dict:
        """검색 품질 등급 평가"""
        score = 0
        max_score = 100

        # 기본 점수 (40점 만점)
        # 처방 정보 비율 (10점)
        prescription_ratio = basic_metrics['처방 정보'] / \
            len(basic_metrics) if len(basic_metrics) > 0 else 0
        score += min(prescription_ratio * 20, 10)

        # 출처 다양성 (10점)
        source_diversity = basic_metrics['출처 다양성']
        score += min(source_diversity * 2, 10)

        # 평균 관련도 (10점)
        avg_relevance = basic_metrics['평균 관련도']
        score += min(avg_relevance * 3, 10)

        # 이론 내용 포함 (10점)
        theory_ratio = basic_metrics['이론 내용'] / \
            len(basic_metrics) if len(basic_metrics) > 0 else 0
        score += min(theory_ratio * 20, 10)

        # 고급 점수 (60점 만점)
        # 내용 타입 다양성 (15점)
        content_type_diversity = advanced_metrics.get('내용 타입 다양성', 0)
        score += min(content_type_diversity * 3, 15)

        # 대분류 다양성 (15점)
        bb_diversity = advanced_metrics.get('대분류 다양성', 0)
        score += min(bb_diversity * 2.5, 15)

        # 고품질 결과 비율 (15점)
        high_quality_text = advanced_metrics.get('고품질 결과 (≥3.0)', '0개 (0.0%)')
        high_quality_percent = float(
            high_quality_text.split('(')[1].split('%')[0])
        score += min(high_quality_percent * 0.15, 15)

        # 직접 매칭률 (15점)
        direct_match_text = advanced_metrics.get('직접 매칭률', '0개 (0.0%)')
        direct_match_percent = float(
            direct_match_text.split('(')[1].split('%')[0])
        score += min(direct_match_percent * 0.15, 15)

        # 등급 결정
        if score >= 85:
            grade = "S (최우수)"
            description = "매우 포괄적이고 정확한 검색 결과"
        elif score >= 70:
            grade = "A (우수)"
            description = "균형잡힌 좋은 검색 결과"
        elif score >= 55:
            grade = "B (양호)"
            description = "적절한 검색 결과, 일부 개선 여지"
        elif score >= 40:
            grade = "C (보통)"
            description = "기본적인 검색 결과, 개선 필요"
        else:
            grade = "D (미흡)"
            description = "검색 결과 품질 개선 필요"

        return {
            'grade': grade,
            'score': score,
            'description': description
        }

    def _get_improvement_suggestions(self, query: str, basic_metrics: Dict, advanced_metrics: Dict) -> List[str]:
        """검색 개선 제안"""
        suggestions = []

        # 출처 다양성 부족
        if basic_metrics['출처 다양성'] < 3:
            suggestions.append("검색어를 더 일반적인 용어로 바꿔보세요 (출처 다양성 향상)")

        # 평균 관련도 낮음
        if basic_metrics['평균 관련도'] < 2.0:
            suggestions.append("더 구체적인 한자 용어를 사용해보세요 (관련도 향상)")

        # 처방 정보 부족 (처방 관련 검색인 경우)
        if ('湯' in query or '散' in query or '丸' in query) and basic_metrics['처방 정보'] < 5:
            suggestions.append("처방명을 정확히 입력하거나 '처방', '치료' 등의 키워드를 추가해보세요")

        # 직접 매칭률 낮음
        direct_match_text = advanced_metrics.get('직접 매칭률', '0개 (0.0%)')
        direct_match_percent = float(
            direct_match_text.split('(')[1].split('%')[0])
        if direct_match_percent < 30:
            suggestions.append("동의어나 관련 용어를 함께 검색해보세요")

        # 내용 타입 다양성 부족
        if advanced_metrics.get('내용 타입 다양성', 0) < 3:
            suggestions.append("더 포괄적인 검색을 위해 관련 증상이나 이론도 함께 검색해보세요")

        # 고품질 결과 비율 낮음
        high_quality_text = advanced_metrics.get('고품질 결과 (≥3.0)', '0개 (0.0%)')
        high_quality_percent = float(
            high_quality_text.split('(')[1].split('%')[0])
        if high_quality_percent < 20:
            suggestions.append("검색어의 정확한 한자 표기를 확인하거나 검색 결과 수를 늘려보세요")

        return suggestions

    def save_search_results(self, query: str, results: List[Dict], answer: str):
        """검색 결과 자동 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_dir = self.save_path / f"{query}_{timestamp}"
        search_dir.mkdir(exist_ok=True)

        result_file = search_dir / f"{query}_{timestamp}.txt"

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"검색어: {query}\n")
            f.write(f"검색 시간: {timestamp}\n")
            f.write(f"검색 결과 수: {len(results)}개\n")
            f.write("=" * 50 + "\n\n")

            f.write("🤖 AI 답변 (근거 문헌 포함):\n")
            f.write(answer + "\n\n")
            f.write("=" * 50 + "\n\n")

            # 관련 검색어 제안도 저장
            categorized_suggestions = self.suggest_related_queries(
                query, results)
            if categorized_suggestions:
                f.write("💡 관련 검색어 제안:\n")
                suggestion_count = 1
                for category, suggestions in categorized_suggestions.items():
                    if suggestions:
                        f.write(f"\n{category}:\n")
                        for suggestion in suggestions:
                            f.write(f"   {suggestion_count}. {suggestion}\n")
                            suggestion_count += 1
                f.write("\n" + "=" * 50 + "\n\n")

            # 검색 품질 메트릭도 저장
            f.write("📊 검색 품질 메트릭:\n")
            basic_metrics = {
                '처방 정보': len([r for r in results if r['metadata'].get('type') == 'prescription']),
                '이론 내용': len([r for r in results if r['metadata'].get('BB')]),
                '출처 다양성': len(set(r['metadata'].get('source_file', 'unknown') for r in results)),
                '평균 관련도': sum(r.get('score', 0) for r in results) / len(results)
            }

            for metric, value in basic_metrics.items():
                if metric == '평균 관련도':
                    f.write(f"   • {metric}: {value:.3f}\n")
                else:
                    f.write(f"   • {metric}: {value}개\n")

            f.write("\n" + "=" * 50 + "\n\n")

            f.write("📚 검색된 원문:\n\n")
            for i, result in enumerate(results):
                f.write(f"[문서 {i + 1}] (유사도: {result['score']:.3f})\n")
                f.write(f"출처: {result['metadata']['source_file']}\n")

                if result['metadata'].get('BB'):
                    f.write(f"대분류: {result['metadata']['BB']}\n")
                if result['metadata'].get('CC'):
                    f.write(f"중분류: {result['metadata']['CC']}\n")
                if result['metadata'].get('prescription_name'):
                    f.write(
                        f"처방명: {result['metadata']['prescription_name']}\n")

                f.write(f"내용:\n{result['content']}\n")
                f.write("-" * 30 + "\n\n")

        print(f"💾 검색 결과 자동 저장 완료: {result_file}")

    def display_search_results(self, query: str, results: List[Dict], answer: str,
                               show_details: bool = False, show_related_queries: bool = True) -> bool:
        """검색 결과 표시 (관련 검색어 제안 포함)"""
        print("\n" + "=" * 50)

        # AI 답변 표시
        if self.llm_manager and self.llm_manager.is_available():
            print("🤖 AI 답변 (근거 문헌 주석 포함):")
            print("-" * 30)
            print(answer)
        else:
            print("⚠️ AI 답변을 생성할 수 없어 검색 결과만 표시합니다.")

        # 자동 저장 실행
        self.save_search_results(query, results, answer)

        print("=" * 50)

        # 카테고리별 검색 결과 표시
        self._display_categorized_results(results)

        # 관련 검색어 제안 표시
        if show_related_queries:
            self.display_related_queries(query, results)

        # 상세 검색 결과 보기 옵션
        if not show_details:
            show_details_input = input(
                "\n📋 모든 검색 결과의 전체 내용을 보시겠습니까? (y/n): ").strip().lower()
            show_details = show_details_input in ['y', 'yes', 'ㅇ', '네', '예']

        if show_details:
            self._display_detailed_results(results)

        return show_details

    def _categorize_results(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """검색 결과를 카테고리별로 분류"""
        categories = defaultdict(list)

        for result in results:
            metadata = result['metadata']

            # 카테고리 결정 로직
            category_info = self._determine_category(result)

            # 각 결과에 표시용 제목과 요약 추가
            enhanced_result = {
                **result,
                'title': category_info['title'],
                'summary': category_info['summary'],
                'category_icon': category_info['icon']
            }

            categories[category_info['category']].append(enhanced_result)

        return dict(categories)

    def _determine_category(self, result: Dict) -> Dict[str, str]:
        """개별 결과의 카테고리 결정"""
        metadata = result['metadata']
        content = result['content']

        # 처방 카테고리
        if (metadata.get('type') == 'prescription' or
            metadata.get('prescription_name') or
                any(keyword in content for keyword in ['湯', '散', '丸', '膏', 'DP'])):

            prescription_name = metadata.get('prescription_name', '처방')
            if not prescription_name or prescription_name == '처방':
                # 내용에서 처방명 추출 시도
                import re
                matches = re.findall(r'([一-龯]{2,6}[湯散丸膏])', content)
                prescription_name = matches[0] if matches else '처방'

            return {
                'category': '💊 처방 및 치료법',
                'title': prescription_name,
                'summary': content[:100] + "..." if len(content) > 100 else content,
                'icon': '💊'
            }

        # 이론/개념 카테고리
        elif (metadata.get('BB') in ['身形', '精', '氣', '神'] or
                any(keyword in content for keyword in ['經曰', '靈樞曰', '內經曰', '理論', '원리'])):

            title = metadata.get('CC', metadata.get('BB', '이론'))
            return {
                'category': '📚 이론 및 개념',
                'title': title,
                'summary': content[:80] + "..." if len(content) > 80 else content,
                'icon': '📚'
            }

        # 병증/증상 카테고리
        elif any(keyword in content for keyword in ['證', '病', '症', '痛', '虛', '實', '寒', '熱']):

            # 병증명 추출
            import re
            symptom_patterns = [
                r'([一-龯]{2,4}[證病症])',
                r'([一-龯]{1,3}[虛實])',
                r'([一-龯]{2,4}[痛])'
            ]

            symptom_name = '병증'
            for pattern in symptom_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    symptom_name = matches[0]
                    break

            return {
                'category': '🩺 병증 및 증상',
                'title': symptom_name,
                'summary': content[:80] + "..." if len(content) > 80 else content,
                'icon': '🩺'
            }

        # 약물/본초 카테고리
        elif any(keyword in content for keyword in ['味', '性', '歸經', '效能', '主治', '용법']):

            # 약물명 추출
            import re
            herb_patterns = [
                r'([一-龯]{2,4}[參芎歸芍地黃])',
                r'([一-龯]{2,4})'
            ]

            herb_name = '약물'
            for pattern in herb_patterns:
                matches = re.findall(pattern, content[:50])  # 앞부분에서만 찾기
                if matches:
                    herb_name = matches[0]
                    break

            return {
                'category': '🌿 약물 및 본초',
                'title': herb_name,
                'summary': content[:80] + "..." if len(content) > 80 else content,
                'icon': '🌿'
            }

        # 진단/맥법 카테고리
        elif any(keyword in content for keyword in ['脈', '診', '辨', '察', '候']):

            return {
                'category': '🔍 진단 및 맥법',
                'title': metadata.get('CC', '진단법'),
                'summary': content[:80] + "..." if len(content) > 80 else content,
                'icon': '🔍'
            }

        # 기타 일반 내용
        else:
            title = metadata.get('CC', metadata.get('BB', '일반'))
            return {
                'category': '📖 기타 내용',
                'title': title,
                'summary': content[:80] + "..." if len(content) > 80 else content,
                'icon': '📖'
            }

    def _display_categorized_results(self, results: List[Dict]):
        """카테고리별로 그룹핑된 검색 결과 표시"""
        categories = self._categorize_results(results)

        print(f"\n📋 검색 결과 분석 ({len(results)}개 문서)")
        print("=" * 60)

        # 카테고리별 표시 순서 정의
        category_order = [
            '💊 처방 및 치료법',
            '🩺 병증 및 증상',
            '📚 이론 및 개념',
            '🌿 약물 및 본초',
            '🔍 진단 및 맥법',
            '📖 기타 내용'
        ]

        total_shown = 0

        for category in category_order:
            items = categories.get(category, [])
            if not items:
                continue

            print(f"\n{category} ({len(items)}개)")
            print("-" * 40)

            # 각 카테고리에서 상위 5개까지 표시
            display_count = min(5, len(items))
            for i, item in enumerate(items[:display_count]):
                score = item.get('score', 0)
                title = item['title']
                summary = item['summary']

                print(f"   {i + 1}. {title}")
                print(f"      관련도: {score:.3f}")
                print(f"      요약: {summary}")

                if i < display_count - 1:
                    print()

            if len(items) > display_count:
                print(f"      ... 외 {len(items) - display_count}개 더")

            total_shown += display_count

        # 전체 통계
        print(f"\n📊 카테고리별 분포:")
        for category in category_order:
            count = len(categories.get(category, []))
            if count > 0:
                percentage = (count / len(results)) * 100
                print(f"   {category}: {count}개 ({percentage:.1f}%)")

        print(f"\n💡 상위 {total_shown}개 결과를 표시했습니다.")

    def _display_detailed_results(self, results: List[Dict]):
        """상세 검색 결과 표시"""
        print("\n" + "=" * 60)
        print("📚 전체 검색 결과 상세 내용:")

        for i, result in enumerate(results):
            print(f"\n[문서 {i + 1}] (유사도: {result['score']:.3f})")
            print(f"출처: {result['metadata']['source_file']}")

            if result['metadata'].get('BB'):
                print(f"대분류: {result['metadata']['BB']}")
            if result['metadata'].get('CC'):
                print(f"중분류: {result['metadata']['CC']}")
            if result['metadata'].get('prescription_name'):
                print(f"처방명: {result['metadata']['prescription_name']}")

            # 카테고리 정보 표시 (추가된 부분)
            if 'title' in result:
                print(
                    f"카테고리: {result.get('category_icon', '📄')} {result['title']}")

            print(f"전체 내용:\n{result['content']}")
            print("-" * 40)

        print("=" * 60)

    def get_continue_choice(self) -> bool:
        """계속 검색할지 선택 (관련 검색어 옵션 추가)"""
        while True:
            continue_search = input(
                "\n🔄 다른 질문을 하시겠습니까? (y/n): ").strip().lower()
            if continue_search in ['y', 'yes', 'ㅇ', '네', '예']:
                return True
            elif continue_search in ['n', 'no', 'ㄴ', '아니오', '아니요']:
                return False
            else:
                print("y 또는 n을 입력해주세요.")

    def display_category_statistics(self, results: List[Dict]):
        """카테고리 통계 표시 (추가 기능)"""
        categories = self._categorize_results(results)

        print("\n📈 검색 결과 상세 통계:")
        print("=" * 40)

        total_results = len(results)

        for category, items in categories.items():
            count = len(items)
            percentage = (count / total_results) * 100
            avg_score = sum(item.get('score', 0)
                            for item in items) / count if count > 0 else 0

            print(f"{category}")
            print(f"  📊 개수: {count}개 ({percentage:.1f}%)")
            print(f"  🎯 평균 관련도: {avg_score:.3f}")

            # 상위 항목들의 제목 표시
            if items:
                top_items = sorted(items, key=lambda x: x.get(
                    'score', 0), reverse=True)[:3]
                titles = [item['title'] for item in top_items]
                print(f"  🏆 주요 항목: {', '.join(titles)}")
            print()

    def export_categorized_results(self, query: str, results: List[Dict], format='txt'):
        """카테고리별 결과를 파일로 내보내기 (추가 기능)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        categories = self._categorize_results(results)

        if format == 'txt':
            export_file = self.save_path / \
                f"{query}_categorized_{timestamp}.txt"

            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(f"동의보감 검색 결과 - 카테고리별 분류\n")
                f.write(f"검색어: {query}\n")
                f.write(f"검색 시간: {timestamp}\n")
                f.write(f"총 결과 수: {len(results)}개\n")
                f.write("=" * 60 + "\n\n")

                for category, items in categories.items():
                    f.write(f"{category} ({len(items)}개)\n")
                    f.write("-" * 40 + "\n")

                    for i, item in enumerate(items):
                        f.write(
                            f"{i + 1}. {item['title']} (관련도: {item['score']:.3f})\n")
                        f.write(f"   {item['summary']}\n")
                        f.write(
                            f"   출처: {item['metadata']['source_file']}\n\n")

                    f.write("\n")

            print(f"📁 카테고리별 결과 저장 완료: {export_file}")

        return export_file if format == 'txt' else None

    def display_search_results_with_metrics(self, query: str, results: List[Dict], answer: str,
                                            show_metrics: bool = True, show_related_queries: bool = True) -> bool:
        """메트릭 포함 검색 결과 표시 (관련 검색어 포함)"""

        # 기존 검색 결과 표시 (관련 검색어 포함)
        show_details = self.display_search_results(
            query, results, answer, show_related_queries=show_related_queries)

        # 검색 품질 메트릭 표시
        if show_metrics:
            self.show_search_metrics(query, results)

        return show_details
