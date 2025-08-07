#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의보감 문서 처리 모듈 - document_processor_improved.py (개선된 버전)
하드코딩된 TCM 용어 사전을 표준한의학용어집 기반으로 교체
문서 로딩, 파싱, 청킹을 담당
"""

import os
import re
import unicodedata
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

try:
    import jieba
    import opencc
    import tiktoken
except ImportError as e:
    print(f"필수 라이브러리가 설치되지 않았습니다: {e}")
    raise


class DocumentProcessor:
    def __init__(self, data_path: str, terms_manager=None):
        """문서 처리기 초기화"""
        self.data_path = Path(data_path)
        self.terms_manager = terms_manager

        # 중국어 처리 도구 초기화
        self.cc = opencc.OpenCC('t2s')

        # 토크나이저 초기화 (GPT-4 호환)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # 동적 용어 사전 (표준용어집 기반)
        self.dynamic_tcm_terms = {}
        self.category_patterns = {}
        self.prescription_patterns = set()
        self.herb_patterns = set()

        # 초기화
        self.setup_chinese_tools()
        self._build_dynamic_tcm_dictionary()

    def setup_chinese_tools(self):
        """중국어 처리 도구 설정 (표준용어집 기반)"""
        print("🔧 중국어 처리 도구 설정 중...")

        try:
            if self.terms_manager and hasattr(self.terms_manager, 'search_index'):
                print("📚 표준한의학용어집 기반 jieba 설정 중...")
                added_count = 0

                # 표준용어집의 모든 용어를 jieba에 추가
                for term in self.terms_manager.search_index.keys():
                    if len(term) >= 2 and self._is_chinese_term(term):
                        jieba.add_word(term, freq=None, tag=None)
                        added_count += 1

                # 추가로 용어집에서 고빈도 용어들 우선 처리
                high_priority_categories = ['처방', '약물', '병증', '생리', '병리']
                for category in high_priority_categories:
                    try:
                        category_terms = self.terms_manager.search_by_category(
                            category, limit=100)
                        for term_data in category_terms:
                            term_name = term_data.get('용어명', '')
                            hanja_name = term_data.get('용어명_한자', '')

                            if term_name and len(term_name) >= 2:
                                jieba.add_word(
                                    term_name, freq=10, tag=category)
                            if hanja_name and len(hanja_name) >= 2:
                                jieba.add_word(
                                    hanja_name, freq=20, tag=category)

                            # 동의어도 추가
                            synonyms = term_data.get('동의어', [])
                            for synonym in synonyms:
                                if synonym and len(synonym) >= 2:
                                    jieba.add_word(
                                        synonym, freq=5, tag=category)
                    except Exception as e:
                        print(f"⚠️ {category} 카테고리 처리 실패: {e}")

                print(f"✅ {added_count}개 표준용어 jieba 등록 완료")

        except Exception as e:
            print(f"⚠️ 표준용어집 기반 jieba 설정 실패: {e}")
            self._setup_fallback_jieba_terms()

    def _is_chinese_term(self, term: str) -> bool:
        """한자 용어인지 확인"""
        if not term:
            return False
        return all('\u4e00' <= char <= '\u9fff' for char in term)

    def _setup_fallback_jieba_terms(self):
        """폴백: 기본 중의학 용어를 jieba에 추가"""
        print("📚 기본 중의학 용어로 jieba 설정 중...")

        basic_terms = [
            # 처방
            "四物湯", "六君子湯", "補中益氣湯", "當歸補血湯", "犀角地黃湯",
            "八物湯", "十全大補湯", "人參養榮湯", "歸脾湯", "甘麥大棗湯",

            # 약재
            "人參", "當歸", "川芎", "白芍", "熟地黃", "生地黃", "黃芪", "白朮",
            "茯苓", "甘草", "陳皮", "半夏", "枳實", "厚朴", "桔梗", "杏仁",
            "麥門冬", "五味子", "山藥", "茯神", "遠志", "石菖蒲", "朱砂", "龍骨",

            # 병증
            "驚悸", "健忘", "癲癇", "眩暈", "失眠", "虛勞", "血虛", "氣虛",
            "陰虛", "陽虛", "脾胃虛弱", "心腎不交", "肝鬱氣滯", "痰濕阻絡",

            # 이론
            "陰陽", "五行", "臟腑", "氣血", "經絡", "精氣神", "君臣佐使",
            "營衛", "三焦", "命門", "元氣", "真陰", "真陽", "先天之本"
        ]

        for term in basic_terms:
            jieba.add_word(term, freq=10)
        print(f"✅ {len(basic_terms)}개 기본 중의학 용어 추가 완료")

    def _build_dynamic_tcm_dictionary(self):
        """표준용어집 기반 동적 TCM 사전 구축"""
        print("🔨 표준용어집 기반 동적 TCM 사전 구축 중...")

        if not self.terms_manager:
            self._build_fallback_tcm_dictionary()
            return

        try:
            # 카테고리별 용어 수집
            categories_mapping = {
                'symptoms': ['병증', '증상', '징후'],
                'herbs': ['약물'],
                'prescriptions': ['처방'],
                'theories': ['생리', '병리', '변증'],
                'methods': ['치법', '침구'],
                'diagnostics': ['진찰']
            }

            self.dynamic_tcm_terms = {}
            total_terms = 0

            for key, categories in categories_mapping.items():
                terms_list = []

                for category in categories:
                    try:
                        category_terms = self.terms_manager.search_by_category(
                            category, limit=200)
                        for term_data in category_terms:
                            # 한자명 우선, 없으면 용어명
                            hanja = term_data.get('용어명_한자', '')
                            hangul = term_data.get('용어명', '')

                            if hanja and len(hanja) >= 2:
                                terms_list.append(hanja)
                            elif hangul and len(hangul) >= 2:
                                terms_list.append(hangul)

                            # 동의어도 포함
                            synonyms = term_data.get('동의어', [])
                            for synonym in synonyms:
                                if synonym and len(synonym) >= 2:
                                    terms_list.append(synonym)

                    except Exception as e:
                        print(f"⚠️ {category} 카테고리 처리 실패: {e}")

                # 중복 제거 및 정리
                self.dynamic_tcm_terms[key] = list(set(terms_list))
                total_terms += len(self.dynamic_tcm_terms[key])

            # 특별 패턴 구축
            self._build_prescription_patterns()
            self._build_herb_patterns()

            print(f"✅ 동적 TCM 사전 구축 완료: 총 {total_terms}개 용어")
            for key, terms in self.dynamic_tcm_terms.items():
                print(f"   {key}: {len(terms)}개")

        except Exception as e:
            print(f"⚠️ 동적 TCM 사전 구축 실패: {e}")
            self._build_fallback_tcm_dictionary()

    def _build_prescription_patterns(self):
        """처방 패턴 동적 구축"""
        try:
            prescriptions = self.terms_manager.search_by_category(
                '처방', limit=300)

            for prescription in prescriptions:
                hanja = prescription.get('용어명_한자', '')
                if hanja:
                    # 처방 접미사 추출
                    for suffix in ['湯', '散', '丸', '膏', '飲', '丹', '方', '子']:
                        if hanja.endswith(suffix):
                            self.prescription_patterns.add(suffix)

                    # 전체 처방명도 패턴으로 추가
                    if len(hanja) >= 3:
                        self.prescription_patterns.add(hanja)

            print(f"📊 처방 패턴 {len(self.prescription_patterns)}개 구축")

        except Exception as e:
            print(f"⚠️ 처방 패턴 구축 실패: {e}")
            self.prescription_patterns = {'湯', '散', '丸', '膏'}

    def _build_herb_patterns(self):
        """약재 패턴 동적 구축"""
        try:
            herbs = self.terms_manager.search_by_category('약물', limit=200)

            for herb in herbs:
                hanja = herb.get('용어명_한자', '')
                if hanja and len(hanja) >= 2:
                    self.herb_patterns.add(hanja)

            print(f"📊 약재 패턴 {len(self.herb_patterns)}개 구축")

        except Exception as e:
            print(f"⚠️ 약재 패턴 구축 실패: {e}")
            self.herb_patterns = {'人參', '當歸', '川芎',
                                  '白芍', '熟地黃', '黃芪', '白朮', '茯苓', '甘草'}

    def _build_fallback_tcm_dictionary(self):
        """폴백: 기본 TCM 용어 사전"""
        print("📚 기본 TCM 용어 사전으로 초기화")

        self.dynamic_tcm_terms = {
            "symptoms": [
                "驚悸", "健忘", "癲癇", "眩暈", "失眠", "虛勞", "血虛", "氣虛", "陰虛", "陽虛",
                "心悸", "不寐", "頭痛", "腹痛", "胸痛", "脅痛", "腰痛", "關節痛", "肌肉痛",
                "發熱", "惡寒", "自汗", "盜汗", "咳嗽", "喘息", "嘔吐", "泄瀉", "便秘"
            ],
            "herbs": [
                "人參", "當歸", "川芎", "白芍", "熟地黃", "生地黃", "黃芪", "白朮", "茯苓", "甘草",
                "陳皮", "半夏", "枳實", "厚朴", "桔梗", "杏仁", "麥門冬", "五味子", "山藥", "茯神",
                "遠志", "石菖蒲", "朱砂", "龍骨", "牡蠣", "酸棗仁", "柏子仁", "阿膠", "地骨皮"
            ],
            "prescriptions": [
                "四物湯", "六君子湯", "補中益氣湯", "當歸補血湯", "犀角地黃湯", "八物湯", "十全大補湯",
                "人參養榮湯", "歸脾湯", "甘麥大棗湯", "逍遙散", "柴胡疏肝散", "平胃散", "二陳湯"
            ],
            "theories": [
                "陰陽", "五行", "臟腑", "氣血", "經絡", "精氣神", "君臣佐使", "營衛", "三焦",
                "命門", "元氣", "真陰", "真陽", "先天之本", "後天之本", "腎間動氣"
            ],
            "methods": [
                "汗法", "吐法", "下法", "和法", "溫法", "淸法", "補法", "消法", "針刺", "艾灸"
            ],
            "diagnostics": [
                "望診", "聞診", "問診", "切診", "四診合參", "八綱辨證", "臟腑辨證", "經絡辨證"
            ]
        }

        self.prescription_patterns = {'湯', '散', '丸', '膏', '飲', '丹', '方'}
        self.herb_patterns = set(self.dynamic_tcm_terms["herbs"])

    def normalize_text(self, text: str) -> Tuple[str, str]:
        """텍스트 정규화 (번체 유지 + 간체 변환)"""
        # 유니코드 정규화
        normalized = unicodedata.normalize('NFKC', text)
        # 공백 정리
        normalized = re.sub(r'[　\s]+', ' ', normalized)
        # 간체 변환 (검색 확장용)
        simplified = self.cc.convert(normalized)
        return normalized, simplified

    def calculate_data_hash(self) -> str:
        """데이터 변경 감지용 해시 계산"""
        hash_md5 = hashlib.md5()
        file_info = []

        for root, dirs, files in os.walk(self.data_path):
            for file in sorted(files):
                if file.endswith('.txt'):
                    file_path = Path(root) / file
                    stat = file_path.stat()
                    file_info.append(f"{file}:{stat.st_mtime}:{stat.st_size}")

        hash_md5.update('|'.join(file_info).encode('utf-8'))
        return hash_md5.hexdigest()

    def load_documents(self) -> List[Dict]:
        """동의보감 문서 로드"""
        print("📚 동의보감 문서 로딩 중...")
        all_chunks = []

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = Path(root) / file
                    print(f"   📄 {file_path.name} 처리 중...")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        chunks = self.parse_document_structure(content)
                        for chunk in chunks:
                            chunk['metadata']['source_file'] = file_path.name
                            chunk['metadata']['source_path'] = str(file_path)

                        all_chunks.extend(chunks)

                    except Exception as e:
                        print(f"⚠️ {file_path.name} 처리 실패: {e}")

        print(f"✅ 총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks

    def parse_document_structure(self, text: str) -> List[Dict]:
        """동의보감 문서 구조 파싱 (개선된 버전)"""
        lines = text.split('\n')
        chunks = []
        current_context = {
            'AA': '',  # 편명/권명
            'XX': '',  # 저자 정보
            'BB': '',  # 대분류
            'CC': '',  # 중분류
            'DD': '',  # 소분류
        }

        current_chunk = {'content': [], 'metadata': {}}
        current_prescription = None
        prescription_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 구조 마커 확인
            if len(line) > 2 and line[:2] in ['AA', 'XX', 'OO', 'ZZ', 'BB', 'CC', 'DD', 'DP', 'SS', 'PP']:
                marker = line[:2]
                content = line[2:].strip()

                # 컨텍스트 업데이트
                if marker in current_context:
                    current_context[marker] = content

                # 청킹 로직 (개선된 버전)
                if marker == 'BB':  # 새로운 대분류
                    if current_prescription:
                        chunks.append(self.create_prescription_chunk(
                            current_prescription, prescription_content, current_context))
                        current_prescription = None
                        prescription_content = []

                    if current_chunk['content']:
                        chunks.append(self.finalize_chunk(
                            current_chunk, current_context))
                    current_chunk = {'content': [
                        content], 'metadata': dict(current_context)}

                elif marker == 'CC':  # 새로운 중분류
                    if current_prescription:
                        chunks.append(self.create_prescription_chunk(
                            current_prescription, prescription_content, current_context))
                        current_prescription = None
                        prescription_content = []

                    if current_chunk['content'] and len(' '.join(current_chunk['content'])) > 50:
                        chunks.append(self.finalize_chunk(
                            current_chunk, current_context))
                    current_chunk = {'content': [
                        content], 'metadata': dict(current_context)}

                elif marker == 'DP':  # 처방 시작
                    if current_prescription:
                        chunks.append(self.create_prescription_chunk(
                            current_prescription, prescription_content, current_context))
                    current_prescription = content
                    prescription_content = []

                elif marker == 'SS':  # 처방 상세
                    if current_prescription:
                        prescription_content.append(content)

                elif marker == 'ZZ':  # 본문 내용
                    if not current_prescription:
                        current_chunk['content'].append(content)
                    else:
                        prescription_content.append(content)

        # 마지막 처리
        if current_prescription:
            chunks.append(self.create_prescription_chunk(
                current_prescription, prescription_content, current_context))
        if current_chunk['content']:
            chunks.append(self.finalize_chunk(current_chunk, current_context))

        return chunks

    def create_prescription_chunk(self, prescription_name: str, prescription_content: List[str], context: Dict) -> Dict:
        """처방 전용 청크 생성 (개선된 키워드 추출)"""
        full_content = f"處方: {prescription_name}\n\n" + \
            '\n'.join(prescription_content)
        normalized_text, simplified_text = self.normalize_text(full_content)
        keywords = self.extract_prescription_keywords_improved(
            prescription_name, prescription_content)

        return {
            'content': normalized_text,
            'content_simplified': simplified_text,
            'metadata': {
                **context,
                'type': 'prescription',
                'prescription_name': prescription_name,
                'keywords': keywords,
                'token_count': len(self.tokenizer.encode(normalized_text)),
                'char_count': len(normalized_text)
            }
        }

    def extract_prescription_keywords_improved(self, prescription_name: str, content: List[str]) -> List[str]:
        """개선된 처방 키워드 추출 (표준용어집 기반)"""
        keywords = [prescription_name]
        full_text = ' '.join(content)

        # 1. 약재명 패턴 매칭 (개선된 정규식)
        herb_patterns = [
            r'([一-龯]{2,4})\s*(?:各|每|用|取|加)',
            r'([一-龯]{2,4})\s*[一二三四五六七八九十百千万]\s*[錢兩分斤升合]',
            r'([一-龯]{2,4})\s*(?:少許|適量|若干)',
        ]

        extracted_herbs = set()
        for pattern in herb_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                if len(match) >= 2:
                    extracted_herbs.add(match)

        # 2. 표준용어집에서 약재 확인 및 추가
        if self.terms_manager:
            try:
                for herb in extracted_herbs:
                    herb_info = self.terms_manager.get_term_info(herb)
                    if herb_info and herb_info.get('분류') == '약물':
                        keywords.append(herb)
                        # 관련 약재도 추가
                        related_herbs = self.terms_manager.get_related_terms(
                            herb)
                        for related in related_herbs[:2]:  # 최대 2개만
                            if related in full_text:
                                keywords.append(related)
            except Exception as e:
                print(f"⚠️ 표준용어집 기반 약재 추출 실패: {e}")

        # 3. 동적 TCM 용어 매칭
        for category, terms in self.dynamic_tcm_terms.items():
            for term in terms:
                if term in full_text and term not in keywords:
                    keywords.append(term)

        # 4. 처방 관련 특수 키워드 추출
        prescription_keywords = self._extract_prescription_specific_keywords(
            full_text)
        keywords.extend(prescription_keywords)

        # 5. 효능/주치 키워드 추출
        efficacy_keywords = self._extract_efficacy_keywords(full_text)
        keywords.extend(efficacy_keywords)

        return list(set(keywords))[:20]  # 중복 제거 및 개수 제한

    def _extract_prescription_specific_keywords(self, text: str) -> List[str]:
        """처방 특수 키워드 추출"""
        keywords = []

        # 처방 구성 관련
        composition_patterns = [
            r'右爲末', r'右剉', r'右件', r'右同', r'右各', r'右爲細末',
            r'水煎服', r'酒調服', r'蜜丸', r'湯調服', r'空心服'
        ]

        for pattern in composition_patterns:
            if re.search(pattern, text):
                keywords.append(pattern)

        # 주치증 키워드
        indication_keywords = ['主治', '治', '療', '主', '用於', '適用']
        for keyword in indication_keywords:
            if keyword in text:
                keywords.append(keyword)

        return keywords

    def _extract_efficacy_keywords(self, text: str) -> List[str]:
        """효능 키워드 추출"""
        keywords = []

        # 효능 관련 키워드
        efficacy_patterns = [
            r'補([一-龯]{1,2})', r'益([一-龯]{1,2})', r'養([一-龯]{1,2})',
            r'淸([一-龯]{1,2})', r'瀉([一-龯]{1,2})', r'溫([一-龯]{1,2})',
            r'涼([一-龯]{1,2})', r'散([一-龯]{1,2})', r'收([一-龯]{1,2})'
        ]

        for pattern in efficacy_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                full_term = pattern.replace('([一-龯]{1,2})', match)
                keywords.append(full_term)

        return keywords

    def finalize_chunk(self, chunk: Dict, context: Dict) -> Dict:
        """청크 최종화 (개선된 키워드 추출)"""
        content_text = ' '.join(chunk['content'])
        normalized_text, simplified_text = self.normalize_text(content_text)
        keywords = self.extract_keywords_improved(normalized_text)

        return {
            'content': normalized_text,
            'content_simplified': simplified_text,
            'metadata': {
                **chunk['metadata'],
                **context,
                'keywords': keywords,
                'token_count': len(self.tokenizer.encode(normalized_text)),
                'char_count': len(normalized_text)
            }
        }

    def extract_keywords_improved(self, text: str) -> List[str]:
        """개선된 키워드 추출 (표준용어집 기반)"""
        keywords = []

        # 1. 표준용어집 기반 전문용어 매칭
        if self.terms_manager:
            try:
                # 직접 매칭
                for term in self.terms_manager.search_index.keys():
                    if len(term) >= 2 and term in text:
                        keywords.append(term)

                # 카테고리별 우선순위 매칭
                priority_categories = ['처방', '약물', '병증', '생리']
                for category in priority_categories:
                    category_terms = self.terms_manager.search_by_category(
                        category, limit=50)
                    for term_data in category_terms:
                        hanja = term_data.get('용어명_한자', '')
                        hangul = term_data.get('용어명', '')

                        if hanja and hanja in text and hanja not in keywords:
                            keywords.append(hanja)
                        if hangul and hangul in text and hangul not in keywords:
                            keywords.append(hangul)

            except Exception as e:
                print(f"⚠️ 표준용어집 기반 키워드 추출 실패: {e}")

        # 2. 동적 TCM 용어 매칭
        for category, terms in self.dynamic_tcm_terms.items():
            for term in terms:
                if term in text and term not in keywords:
                    keywords.append(term)

        # 3. jieba 분할 결과 추가 (의미있는 용어만)
        try:
            words = jieba.lcut(text)
            for word in words:
                if (len(word) >= 2 and
                    word not in keywords and
                        self._is_meaningful_medical_term(word)):
                    keywords.append(word)
        except Exception as e:
            print(f"⚠️ jieba 분할 실패: {e}")

        # 4. 특수 패턴 매칭
        special_keywords = self._extract_special_pattern_keywords(text)
        keywords.extend(special_keywords)

        return list(set(keywords))[:15]  # 중복 제거 및 개수 제한

    def _is_meaningful_medical_term(self, word: str) -> bool:
        """의미있는 의학 용어인지 판단"""
        if len(word) < 2:
            return False

        # 한자 비율 확인
        chinese_char_count = sum(
            1 for char in word if '\u4e00' <= char <= '\u9fff')
        if chinese_char_count / len(word) < 0.5:
            return False

        # 의학 관련 특징 확인
        medical_indicators = [
            '病', '症', '證', '痛', '虛', '實', '熱', '寒', '濕', '燥',
            '補', '益', '養', '淸', '瀉', '溫', '涼', '散', '收',
            '湯', '散', '丸', '膏', '飲', '丹', '方'
        ]

        # 동적 패턴과 비교
        if any(indicator in word for indicator in medical_indicators):
            return True

        # 표준용어집에 있는지 확인
        if self.terms_manager:
            try:
                return word in self.terms_manager.search_index
            except:
                pass

        return False

    def _extract_special_pattern_keywords(self, text: str) -> List[str]:
        """특수 패턴 키워드 추출"""
        keywords = []

        # 1. 처방 관련 패턴
        for pattern in self.prescription_patterns:
            if pattern in text:
                keywords.append(pattern)

        # 2. 약재 관련 패턴
        herb_in_text = [herb for herb in self.herb_patterns if herb in text]
        keywords.extend(herb_in_text[:5])  # 최대 5개

        # 3. 수량 표현과 함께 나오는 용어
        quantity_patterns = [
            r'([一-龯]{2,4})\s*[一二三四五六七八九十百千万]\s*[錢兩分斤升合]',
            r'([一-龯]{2,4})\s*各\s*[一二三四五六七八九十]',
            r'([一-龯]{2,4})\s*(?:少許|適量|若干)'
        ]

        for pattern in quantity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2 and self._is_meaningful_medical_term(match):
                    keywords.append(match)

        # 4. 병증 관련 특수 패턴
        symptom_patterns = [
            r'([一-龯]{2,4}[證病症])',
            r'([一-龯]{1,3}[虛實])',
            r'([一-龯]{2,4}[痛])',
            r'([一-龯]{2,4}[熱寒])'
        ]

        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        return keywords

    def analyze_document_statistics(self, chunks: List[Dict]) -> Dict:
        """문서 통계 분석 (개선된 버전)"""
        stats = {
            'total_chunks': len(chunks),
            'total_characters': 0,
            'total_tokens': 0,
            'content_types': defaultdict(int),
            'categories': defaultdict(int),
            'keyword_frequency': Counter(),
            'prescription_count': 0,
            'average_chunk_size': 0,
            'source_distribution': defaultdict(int)
        }

        for chunk in chunks:
            content = chunk['content']
            metadata = chunk['metadata']

            # 기본 통계
            stats['total_characters'] += len(content)
            stats['total_tokens'] += metadata.get('token_count', 0)

            # 내용 타입 분석
            content_type = metadata.get('type', 'general')
            stats['content_types'][content_type] += 1

            # 카테고리 분석
            bb_category = metadata.get('BB', 'unknown')
            cc_category = metadata.get('CC', 'unknown')
            stats['categories'][bb_category] += 1

            # 처방 개수
            if content_type == 'prescription':
                stats['prescription_count'] += 1

            # 키워드 빈도
            keywords = metadata.get('keywords', [])
            for keyword in keywords:
                stats['keyword_frequency'][keyword] += 1

            # 출처 분포
            source_file = metadata.get('source_file', 'unknown')
            stats['source_distribution'][source_file] += 1

        # 평균 청크 크기
        if stats['total_chunks'] > 0:
            stats['average_chunk_size'] = stats['total_characters'] / \
                stats['total_chunks']

        return stats

    def validate_chunks(self, chunks: List[Dict]) -> Dict:
        """청크 유효성 검증"""
        validation_result = {
            'valid_chunks': 0,
            'invalid_chunks': 0,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        for i, chunk in enumerate(chunks):
            try:
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                # 필수 필드 검증
                if not content:
                    validation_result['errors'].append(f"청크 {i}: 내용이 비어있음")
                    validation_result['invalid_chunks'] += 1
                    continue

                if not metadata:
                    validation_result['warnings'].append(f"청크 {i}: 메타데이터가 없음")

                # 내용 길이 검증
                if len(content) < 10:
                    validation_result['warnings'].append(
                        f"청크 {i}: 내용이 너무 짧음 ({len(content)}자)")
                elif len(content) > 2000:
                    validation_result['warnings'].append(
                        f"청크 {i}: 내용이 너무 김 ({len(content)}자)")

                # 키워드 검증
                keywords = metadata.get('keywords', [])
                if not keywords:
                    validation_result['warnings'].append(f"청크 {i}: 키워드가 없음")
                elif len(keywords) > 20:
                    validation_result['warnings'].append(
                        f"청크 {i}: 키워드가 너무 많음 ({len(keywords)}개)")

                # 처방 청크 특별 검증
                if metadata.get('type') == 'prescription':
                    prescription_name = metadata.get('prescription_name')
                    if not prescription_name:
                        validation_result['errors'].append(
                            f"청크 {i}: 처방 청크에 처방명이 없음")
                        validation_result['invalid_chunks'] += 1
                        continue

                validation_result['valid_chunks'] += 1

            except Exception as e:
                validation_result['errors'].append(f"청크 {i}: 검증 중 오류 - {e}")
                validation_result['invalid_chunks'] += 1

        # 권장사항 생성
        total_chunks = len(chunks)
        if total_chunks > 0:
            error_rate = validation_result['invalid_chunks'] / total_chunks
            warning_rate = len(validation_result['warnings']) / total_chunks

            if error_rate > 0.05:
                validation_result['recommendations'].append(
                    "오류율이 5%를 초과합니다. 문서 파싱 로직을 검토하세요.")

            if warning_rate > 0.2:
                validation_result['recommendations'].append(
                    "경고율이 20%를 초과합니다. 청킹 전략을 조정하세요.")

        return validation_result

    def optimize_chunks_for_model(self, chunks: List[Dict], target_model: str = 'gpt-4o-mini') -> List[Dict]:
        """모델에 최적화된 청크 조정"""
        model_configs = {
            'gpt-4o-mini': {
                'max_tokens': 8192,
                'optimal_chunk_tokens': 500,
                'max_chunk_tokens': 1200
            },
            'gpt-4': {
                'max_tokens': 32768,
                'optimal_chunk_tokens': 800,
                'max_chunk_tokens': 2000
            }
        }

        config = model_configs.get(target_model, model_configs['gpt-4o-mini'])
        optimized_chunks = []

        for chunk in chunks:
            content = chunk['content']
            metadata = chunk['metadata']
            token_count = metadata.get(
                'token_count', len(self.tokenizer.encode(content)))

            if token_count <= config['max_chunk_tokens']:
                # 적정 크기 청크는 그대로 유지
                optimized_chunks.append(chunk)
            else:
                # 큰 청크는 분할
                split_chunks = self._split_large_chunk(
                    chunk, config['optimal_chunk_tokens'])
                optimized_chunks.extend(split_chunks)

        print(f"✅ {target_model}에 최적화: {len(chunks)}개 → {len(optimized_chunks)}개 청크")
        return optimized_chunks

    def _split_large_chunk(self, chunk: Dict, target_tokens: int) -> List[Dict]:
        """큰 청크 분할"""
        content = chunk['content']
        metadata = chunk['metadata']

        # 문장 단위로 분할
        sentences = re.split(r'[。．！？\n]', content)
        current_chunk_sentences = []
        current_tokens = 0
        split_chunks = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens > target_tokens and current_chunk_sentences:
                # 현재 청크 저장
                chunk_content = ''.join(current_chunk_sentences)
                normalized_text, simplified_text = self.normalize_text(
                    chunk_content)
                keywords = self.extract_keywords_improved(normalized_text)

                split_chunk = {
                    'content': normalized_text,
                    'content_simplified': simplified_text,
                    'metadata': {
                        **metadata,
                        'keywords': keywords,
                        'token_count': len(self.tokenizer.encode(normalized_text)),
                        'char_count': len(normalized_text),
                        'split_index': len(split_chunks)
                    }
                }
                split_chunks.append(split_chunk)

                # 새 청크 시작
                current_chunk_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # 마지막 청크 처리
        if current_chunk_sentences:
            chunk_content = ''.join(current_chunk_sentences)
            normalized_text, simplified_text = self.normalize_text(
                chunk_content)
            keywords = self.extract_keywords_improved(normalized_text)

            split_chunk = {
                'content': normalized_text,
                'content_simplified': simplified_text,
                'metadata': {
                    **metadata,
                    'keywords': keywords,
                    'token_count': len(self.tokenizer.encode(normalized_text)),
                    'char_count': len(normalized_text),
                    'split_index': len(split_chunks)
                }
            }
            split_chunks.append(split_chunk)

        return split_chunks

    def export_keywords_to_terms_manager(self, chunks: List[Dict]) -> Dict:
        """추출된 키워드를 용어집 관리자로 내보내기"""
        if not self.terms_manager:
            return {'status': 'error', 'message': '용어집 관리자가 연결되지 않았습니다.'}

        keyword_stats = Counter()
        new_terms = []

        for chunk in chunks:
            keywords = chunk['metadata'].get('keywords', [])
            for keyword in keywords:
                keyword_stats[keyword] += 1

        # 고빈도 키워드 중 용어집에 없는 것들 찾기
        for keyword, frequency in keyword_stats.most_common(100):
            if frequency >= 3 and len(keyword) >= 2:  # 최소 3회 이상 등장
                try:
                    if not self.terms_manager.get_term_info(keyword):
                        new_terms.append({
                            'term': keyword,
                            'frequency': frequency,
                            'category': self._infer_term_category(keyword)
                        })
                except:
                    continue

        return {
            'status': 'success',
            'total_keywords': len(keyword_stats),
            'new_terms_found': len(new_terms),
            'new_terms': new_terms[:20]  # 상위 20개만 반환
        }

    def _infer_term_category(self, term: str) -> str:
        """용어 카테고리 추정"""
        # 패턴 기반 카테고리 추정
        if any(suffix in term for suffix in ['湯', '散', '丸', '膏', '飲']):
            return '처방'
        elif any(suffix in term for suffix in ['證', '病', '症']):
            return '병증'
        elif any(suffix in term for suffix in ['虛', '實', '熱', '寒']):
            return '병증'
        elif term in self.herb_patterns:
            return '약물'
        elif any(concept in term for concept in ['陰陽', '五行', '氣血', '經絡']):
            return '이론'
        else:
            return '기타'

    def get_processing_statistics(self) -> Dict:
        """처리 통계 정보 반환"""
        return {
            'terms_manager_connected': self.terms_manager is not None,
            'dynamic_terms_count': sum(len(terms) for terms in self.dynamic_tcm_terms.values()),
            'prescription_patterns_count': len(self.prescription_patterns),
            'herb_patterns_count': len(self.herb_patterns),
            'fallback_mode': not bool(self.terms_manager),
            'jieba_words_added': True,
            'categories': list(self.dynamic_tcm_terms.keys())
        }

    def rebuild_dynamic_dictionary(self):
        """동적 사전 재구축"""
        print("🔄 동적 TCM 사전 재구축 중...")
        self.dynamic_tcm_terms.clear()
        self.prescription_patterns.clear()
        self.herb_patterns.clear()

        self._build_dynamic_tcm_dictionary()
        print("✅ 동적 TCM 사전 재구축 완료")

    def clear_jieba_cache(self):
        """jieba 캐시 초기화"""
        try:
            jieba.dt.cache_file = None
            print("✅ jieba 캐시 초기화 완료")
        except Exception as e:
            print(f"⚠️ jieba 캐시 초기화 실패: {e}")

    def update_terms_manager(self, new_terms_manager):
        """용어집 관리자 업데이트"""
        old_manager = self.terms_manager
        self.terms_manager = new_terms_manager

        if new_terms_manager != old_manager:
            print("🔄 새로운 용어집 관리자로 업데이트 중...")
            self.setup_chinese_tools()
            self._build_dynamic_tcm_dictionary()
            print("✅ 용어집 관리자 업데이트 완료")

    def validate_terms_manager_connection(self) -> bool:
        """용어집 관리자 연결 유효성 검증"""
        if not self.terms_manager:
            return False

        try:
            # 기본 기능 테스트
            test_result = self.terms_manager.get_term_info('血虛')
            return test_result is not None
        except Exception as e:
            print(f"⚠️ 용어집 관리자 연결 검증 실패: {e}")
            return False

    def get_recommended_settings(self) -> Dict:
        """권장 설정 반환"""
        return {
            'chunk_strategy': 'hierarchical',
            'max_chunk_tokens': 1200,
            'optimal_chunk_tokens': 500,
            'enable_terms_manager': True,
            'extract_prescriptions': True,
            'extract_detailed_keywords': True,
            'normalize_traditional_chinese': True,
            'split_large_chunks': True,
            'validate_chunks': True
        }

# 편의 함수들


def create_document_processor(data_path: str, terms_manager=None) -> DocumentProcessor:
    """문서 처리기 생성 편의 함수"""
    return DocumentProcessor(data_path=data_path, terms_manager=terms_manager)


def test_document_processor():
    """테스트용 함수"""
    print("🧪 문서 처리기 테스트")

    # 테스트용 경로 (실제 환경에 맞게 수정)
    test_data_path = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG"

    processor = DocumentProcessor(data_path=test_data_path)

    # 처리 통계 출력
    stats = processor.get_processing_statistics()
    print(f"📊 처리 통계:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 동적 용어 사전 정보
    print(f"\n📚 동적 TCM 용어 사전:")
    for category, terms in processor.dynamic_tcm_terms.items():
        print(f"   {category}: {len(terms)}개 용어")
        if terms:
            print(f"      예시: {', '.join(terms[:5])}")


if __name__ == "__main__":
    test_document_processor()
