#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
답변 생성 모듈 - answer_generator.py (관련 검색어 제안 기능 개선)
LLM을 이용한 답변 생성과 결과 저장을 담당
검색 결과 그룹핑, 카테고리별 표시 및 품질 메트릭 분석 기능
관련 검색어 제안 기능 추가 - 처방명/병증 추출 로직 개선
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
    def __init__(self, llm_manager=None, save_path: str = "/Users/radi/Projects/langchainDATA/Results/DYBGsearch"):
        """답변 생성기 초기화"""
        self.llm_manager = llm_manager
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # 개선된 관련 검색어 추출을 위한 패턴들
        self.prescription_patterns = [
            r'([一-龯]{2,8}[湯散丸膏元丹])',  # 전형적인 처방 접미사
            r'([一-龯]{2,6}飮)',            # ~飮 (예: 四物飮)
            r'([一-龯]{2,6}方)',            # ~방 (예: 逍遙方)
        ]

        # 개선: 병증과 치료법을 구분하는 패턴들
        self.symptom_patterns = [
            r'([一-龯]{2,4}[證病症])',      # 명확한 병증 패턴
            r'([一-龯]{1,3}[虛實])',       # 허실 패턴
            r'([一-龯]{2,4}[痛])',         # 통증 패턴
            r'([一-龯]{2,4}[熱寒濕燥])',   # 한열습조 패턴
            r'([一-龯]{2,4}[脹滿悶])',     # 창만민 패턴
        ]

        # 치료법 패턴 (병증과 구분용)
        self.treatment_patterns = [
            r'治([一-龯]{1,4})',           # 治~ 패턴
            r'主治([一-龯]{2,6})',         # 主治~ 패턴
            r'療([一-龯]{2,4})',           # 療~ 패턴
        ]

        self.herb_patterns = [
            r'([一-龯]{2,4}[參芎歸芍地黃茯苓芪朮])',  # 약재명 패턴
            r'([一-龯]{2,4})',                        # 일반 약재명
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

    def suggest_related_queries(self, query: str, results: List[Dict], max_suggestions: int = 8) -> Dict[str, List[str]]:
        """관련 검색어 제안 기능 (개선된 버전)"""
        # 1. 검색 결과에서 처방명 추출 (개선된 로직)
        prescriptions = self._extract_prescriptions_from_results_improved(
            results)

        # 2. 관련 증상/병증 추출 (개선된 로직)
        symptoms = self._extract_symptoms_from_results_improved(results)

        # 3. 관련 약재 추출
        herbs = self._extract_herbs_from_results(results)

        # 4. 관련 이론/개념 추출
        concepts = self._extract_concepts_from_results(results)

        # 5. 컨텍스트 기반 추천
        contextual_suggestions = self._get_contextual_suggestions(
            query, results)

        # 개선된 분류 로직으로 카테고리별 제안 구성
        suggestion_categories = [
            ("🔥 핵심 처방", [
             p for p in prescriptions if self._is_actual_prescription_improved(p)][:3]),
            ("🩺 관련 병증", [s for s in symptoms if self._is_actual_symptom(s)][:3]),
            ("💊 주요 약재", [h for h in herbs if self._is_herb_name(h)][:3]),
            ("📚 관련 개념", concepts[:2]),
            ("🎯 맞춤 제안", contextual_suggestions[:2])
        ]

        # 카테고리별로 제안사항 수집
        categorized_suggestions = {}
        all_suggestions = []

        for category, items in suggestion_categories:
            if items:
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

    def _extract_prescriptions_from_results_improved(self, results: List[Dict]) -> List[str]:
        """개선된 처방명 추출 (문제 1 해결)"""
        prescription_counts = Counter()

        for result in results:
            # 1순위: 메타데이터의 DP 레벨 처방명 (가장 확실)
            if result['metadata'].get('prescription_name'):
                dp_prescription = result['metadata']['prescription_name']
                # DP 레벨은 높은 가중치
                prescription_counts[dp_prescription] += 5

            # 2순위: 내용에서 처방 패턴 매칭 (개선된 로직)
            content = result['content']

            # 개선된 처방 패턴으로 추출
            for pattern in self.prescription_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) >= 3:  # 최소 3글자 이상
                        prescription_counts[match] += 2

            # 추가: '~子' 처방 패턴 (DP 레벨에서 나오는 경우만)
            # DP 마커 근처에 있는 '~子'만 처방으로 인정
            if 'DP' in content:
                dp_sections = re.split(r'DP', content)
                for section in dp_sections[1:]:  # DP 이후 섹션들만
                    # DP 직후 50자 이내에서 ~子 패턴 찾기
                    first_part = section[:50]
                    zi_matches = re.findall(r'([一-龯]{2,4}子)', first_part)
                    for match in zi_matches:
                        # 처방 context 확인 (湯, 散, 丸 등이 같이 언급되거나 처방 설명이 있는 경우)
                        if self._has_prescription_context(section[:200], match):
                            prescription_counts[match] += 3

        # 빈도순으로 정렬하여 상위 항목 추출
        raw_prescriptions = [
            name for name, count in prescription_counts.most_common(20) if count >= 2]

        # 개선된 필터링 적용
        filtered_prescriptions = []
        for prescription in raw_prescriptions:
            if (self._is_meaningful_prescription_term(prescription) and
                    self._is_actual_prescription_improved(prescription)):
                filtered_prescriptions.append(prescription)

        print(
            f"🔍 처방 추출: {len(raw_prescriptions)}개 → 필터링 후 {len(filtered_prescriptions)}개")
        return filtered_prescriptions[:6]

    def _has_prescription_context(self, text: str, term: str) -> bool:
        """처방 컨텍스트 확인 (처방인지 단순 약재인지 구분)"""
        # 처방 관련 키워드가 근처에 있는지 확인
        prescription_keywords = [
            '治', '主治', '方', '湯', '散', '丸', '膏', '飮', '丹', '元',
            '服', '煎', '用法', '분량', '右', '右爲末', '右剉'
        ]

        # term 주변 50자 내에서 처방 키워드 찾기
        term_pos = text.find(term)
        if term_pos == -1:
            return False

        start = max(0, term_pos - 25)
        end = min(len(text), term_pos + len(term) + 25)
        context = text[start:end]

        return any(keyword in context for keyword in prescription_keywords)

    def _is_actual_prescription_improved(self, term: str) -> bool:
        """개선된 실제 처방명 판단 (문제 1 해결)"""
        # 1. 전형적인 처방 패턴
        import re
        prescription_patterns = [
            r'.*[湯散丸膏元丹]$',  # 전형적인 처방 형태
            r'.*飮$',              # ~飮
            r'.*方$',              # ~방
        ]

        for pattern in prescription_patterns:
            if re.match(pattern, term):
                return True

        # 2. '~子' 처방의 경우: 길이와 컨텍스트로 판단
        if term.endswith('子'):
            # 3글자 이상이고 알려진 처방명이면 허용
            if len(term) >= 3:
                known_zi_prescriptions = [
                    '逍遙子', '甘露子', '紫雪子', '至寶子', '牛黃子',
                    '局方子', '金櫃子', '千金子', '外科子'
                ]
                # 정확한 매칭 또는 패턴 매칭
                return term in known_zi_prescriptions or len(term) >= 4

        # 3. 알려진 처방명 리스트 (추가 확인)
        known_prescriptions = [
            '小建中湯', '二陳湯', '四君子湯', '四物湯', '六君子湯', '補中益氣湯',
            '當歸補血湯', '犀角地黃湯', '安神定志丸', '天王補心丹', '朱砂安神丸'
        ]

        return term in known_prescriptions

    def _extract_symptoms_from_results_improved(self, results: List[Dict]) -> List[str]:
        """개선된 증상/병증 추출 (문제 2 해결)"""
        symptom_counts = Counter()
        treatment_extracted_symptoms = Counter()  # 치료법에서 추출된 병증들

        for result in results:
            content = result['content']

            # 1. 직접적인 병증 패턴 매칭
            for pattern in self.symptom_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) >= 2:
                        symptom_counts[match] += 2

            # 2. 치료법 패턴에서 병증 추출 (개선된 로직)
            for pattern in self.treatment_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) >= 2 and self._is_valid_symptom_from_treatment(match):
                        # 치료법에서 추출된 것은 별도 카운트 (가중치 낮음)
                        treatment_extracted_symptoms[match] += 1

            # 3. 특정 키워드 기반 추출 (명확한 병증만)
            clear_symptom_keywords = [
                '驚悸', '健忘', '眩暈', '失眠', '虛勞', '血虛', '氣虛', '陰虛', '陽虛',
                '脾胃虛', '心悸', '不寐', '頭痛', '腹痛', '胸痛', '癲癇', '中風', '咳嗽',
                '哮喘', '泄瀉', '便秘', '黃疸', '水腫', '淋病', '崩漏', '帶下'
            ]
            for keyword in clear_symptom_keywords:
                if keyword in content:
                    symptom_counts[keyword] += 3

        # 치료법에서 추출된 병증들을 메인 카운트에 추가 (낮은 가중치)
        for symptom, count in treatment_extracted_symptoms.items():
            # 이미 직접 추출된 병증이 아닌 경우만 추가
            if symptom not in symptom_counts:
                symptom_counts[symptom] += count * 0.5  # 가중치 절반

        # 빈도순 정렬 후 필터링 적용
        raw_symptoms = [symptom for symptom,
                        count in symptom_counts.most_common(15) if count >= 1.5]

        # 개선된 필터링: 실제 병증만 선별
        filtered_symptoms = []
        for symptom in raw_symptoms:
            if self._is_actual_symptom(symptom):
                filtered_symptoms.append(symptom)

        print(
            f"🔍 증상 추출: {len(raw_symptoms)}개 → 필터링 후 {len(filtered_symptoms)}개")
        return filtered_symptoms[:8]

    def _is_valid_symptom_from_treatment(self, term: str) -> bool:
        """치료법에서 추출된 용어가 유효한 병증인지 판단"""
        # 치료 동작이 아닌 실제 병증인지 확인
        invalid_treatment_terms = [
            '病', '症', '疾', '患', '療', '癒', '愈', '效', '驗', '止', '除', '去', '消',
            '散', '解', '破', '下', '上', '中', '內', '外', '表', '裏', '虛', '實'
        ]

        # 단일 글자이거나 치료 동작 용어면 제외
        if len(term) == 1 or term in invalid_treatment_terms:
            return False

        # 명확한 병증 접미사가 있으면 허용
        valid_symptom_suffixes = ['虛', '實', '熱',
                                  '寒', '濕', '燥', '痛', '悸', '暈', '眠']
        if any(term.endswith(suffix) for suffix in valid_symptom_suffixes):
            return True

        # 알려진 병증명이면 허용
        known_symptoms = [
            '간허', '신허', '기허', '혈허', '심허', '폐허', '비허', '위허',
            '간열', '심열', '폐열', '위열', '간기울결', '심기부족'
        ]

        return term in known_symptoms or len(term) >= 2

    def _is_actual_symptom(self, term: str) -> bool:
        """실제 병증인지 판단 (개선된 로직)"""
        # 치료법 관련 용어들 제외
        treatment_terms = [
            '治虛', '治實', '治熱', '治寒', '治痛', '治病', '治療', '主治',
            '療虛', '療實', '補虛', '瀉實', '清熱', '溫寒', '止痛',
            '治法', '用法', '服法', '煎法'
        ]

        if term in treatment_terms:
            return False

        # 명확한 병증 패턴들
        symptom_patterns = [
            r'.*[虛實]$',      # ~허, ~실
            r'.*[痛]$',        # ~통
            r'.*[熱寒]$',      # ~열, ~한
            r'.*[證病症]$',    # ~증, ~병, ~증
            r'.*[悸暈眠]$',    # ~계, ~훈, ~면
        ]

        import re
        for pattern in symptom_patterns:
            if re.match(pattern, term):
                return True

        # 알려진 명확한 병증명들
        clear_symptoms = [
            '驚悸', '健忘', '眩暈', '失眠', '虛勞', '血虛', '氣虛', '陰虛', '陽虛',
            '心悸', '不寐', '頭痛', '腹痛', '胸痛', '癲癇', '中風', '咳嗽', '哮喘',
            '泄瀉', '便秘', '黃疸', '水腫', '崩漏', '帶下', '遺精', '陽痿', '早泄'
        ]

        return term in clear_symptoms

    def _extract_herbs_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 약재명 추출"""
        herb_counts = Counter()

        # 주요 약재 리스트
        major_herbs = [
            '人參', '當歸', '川芎', '白芍', '熟地黃', '生地黃', '黃芪', '白朮',
            '茯苓', '甘草', '陳皮', '半夏', '枳實', '厚朴', '桔梗', '杏仁',
            '麥門冬', '五味子', '山藥', '茯神', '遠志', '石菖蒲', '朱砂', '龍骨',
            '牡蠣', '酸棗仁', '柏子仁', '阿膠', '地骨皮', '知母', '黃柏', '山茱萸'
        ]

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

        # 빈도순 정렬 후 필터링 적용
        raw_herbs = [herb for herb,
                     count in herb_counts.most_common(15) if count >= 2]
        filtered_herbs = self._filter_prescription_suggestions(raw_herbs)

        print(f"🔍 약재 추출: {len(raw_herbs)}개 → 필터링 후 {len(filtered_herbs)}개")
        return filtered_herbs[:8]

    def _extract_concepts_from_results(self, results: List[Dict]) -> List[str]:
        """검색 결과에서 이론/개념 추출"""
        concept_counts = Counter()

        # 중의학 핵심 개념들
        key_concepts = [
            '陰陽', '五行', '臟腑', '氣血', '經絡', '精氣神', '君臣佐使',
            '四象', '八綱', '六經', '營衛', '三焦', '命門', '元氣', '真陰',
            '火神', '溫補', '滋陰', '理氣', '活血', '化痰', '祛濕', '清熱'
        ]

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

        # 빈도순 정렬 후 필터링 적용
        raw_concepts = [concept for concept,
                        count in concept_counts.most_common(10) if count >= 2]
        filtered_concepts = self._filter_prescription_suggestions(raw_concepts)

        print(
            f"🔍 개념 추출: {len(raw_concepts)}개 → 필터링 후 {len(filtered_concepts)}개")
        return filtered_concepts[:5]

    def _get_contextual_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """컨텍스트 기반 맞춤 제안"""
        suggestions = []

        # 1. 쿼리 유형 분석 기반 제안
        if '虛' in query:
            suggestions.extend(['補益', '溫陽', '滋陰', '益氣'])
        elif '湯' in query:
            suggestions.extend(['加減방', '배합금기', '용법용량'])
        elif '病' in query or '證' in query:
            suggestions.extend(['治療방법', '감별진단', '병리기전'])
        elif any(herb in query for herb in ['人參', '當歸', '川芎']):
            suggestions.extend(['약성', '귀경', '효능주치', '배합'])

        # 2. 검색 결과 메타데이터 기반 제안
        for result in results:
            source_file = result['metadata'].get('source_file', '')
            bb = result['metadata'].get('BB', '')

            if source_file:
                if '내경편' in source_file:
                    suggestions.append('정신요법')
                elif '외형편' in source_file:
                    suggestions.append('침구치료')
                elif '잡병편' in source_file:
                    suggestions.append('임상응용')
                elif '탕액편' in source_file:
                    suggestions.append('본초학')

            if bb:
                if bb == '血':
                    suggestions.extend(['補血', '活血', '止血'])
                elif bb == '氣':
                    suggestions.extend(['補氣', '理氣', '降氣'])
                elif bb == '精':
                    suggestions.extend(['補腎', '固精', '滋陰'])

        # 3. 빈도 기반 필터링
        suggestion_counts = Counter(suggestions)
        raw_suggestions = [suggestion for suggestion,
                           count in suggestion_counts.most_common(8)]
        filtered_suggestions = self._filter_prescription_suggestions(
            raw_suggestions)

        print(
            f"🔍 맥락 제안: {len(raw_suggestions)}개 → 필터링 후 {len(filtered_suggestions)}개")
        return filtered_suggestions[:3]

    def _is_herb_name(self, term: str) -> bool:
        """약재명인지 판단"""
        common_herbs = [
            '人參', '當歸', '川芎', '白芍', '熟地黃', '生地黃', '黃芪', '白朮',
            '茯苓', '甘草', '陳皮', '半夏', '枳實', '厚朴', '桔梗', '杏仁',
            '麥門冬', '五味子', '山藥', '茯神', '遠志', '石菖蒲', '朱砂', '龍骨',
            '牡蠣', '酸棗仁', '柏子仁', '阿膠', '地骨皮', '知母', '黃柏', '山茱萸',
            '桂枝', '麻黃', '附子', '乾薑', '細辛', '防風', '荊芥', '薄荷',
            '木通', '車前子', '滑石', '琥珀', '赤茯苓', '澤瀉', '猪苓'
        ]
        return term in common_herbs

    def _is_meaningful_prescription_term(self, term: str) -> bool:
        """처방명으로 의미있는 용어인지 판단"""
        if len(term) < 2:
            return False

        # 제외할 패턴들
        exclude_patterns = [
            r'^右.*', r'.*爲末$', r'.*剉$', r'.*煎服$', r'.*調下$',
            r'^空心.*', r'^每服.*', r'^各.*', r'.*錢$', r'.*兩$',
            r'^《.*》$', r'^[一二三四五六七八九십]+[錢兩分]$',
            r'^[治療用服取入加或及同以]$'
        ]

        for pattern in exclude_patterns:
            if re.match(pattern, term):
                return False

        return True

    def _filter_prescription_suggestions(self, suggestions: List[str]) -> List[str]:
        """무의미한 용어들 필터링"""
        filtered = []
        for suggestion in suggestions:
            if self._is_meaningful_prescription_term(suggestion):
                filtered.append(suggestion)
        return filtered

    def _is_too_similar_to_query(self, query: str, suggestion: str) -> bool:
        """제안어가 검색어와 너무 유사한지 확인"""
        if query in suggestion or suggestion in query:
            return True

        common_chars = set(query) & set(suggestion)
        similarity_ratio = len(common_chars) / \
            max(len(set(query)), len(set(suggestion)))
        return similarity_ratio > 0.8

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

        all_suggestions = []
        for suggestions in categorized_suggestions.values():
            all_suggestions.extend(suggestions)

        if not all_suggestions:
            return None

        while True:
            try:
                choice = input(
                    "\n🤔 선택하세요 (번호 입력 또는 새 검색어 입력, Enter로 건너뛰기): ").strip()

                if not choice:
                    return None

                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(all_suggestions):
                        selected_query = all_suggestions[choice_num - 1]
                        print(f"✅ '{selected_query}'를 선택했습니다.")
                        return selected_query
                    else:
                        print(f"❌ 1~{len(all_suggestions)} 범위의 번호를 입력해주세요.")
                        continue
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

    # 기존 메서드들은 그대로 유지...
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

        # 🆕 관련 검색어 제안 표시
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
