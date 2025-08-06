#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의보감 문서 처리 모듈 - document_processor.py
문서 로딩, 파싱, 청킹을 담당
"""

import os
import re
import unicodedata
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
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
        self.setup_chinese_tools()

        # 토크나이저 초기화 (GPT-4 호환)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # 중의학 전문 용어 사전 (기본 백업용)
        self.tcm_terms = self.load_basic_tcm_terms()

    def setup_chinese_tools(self):
        """중국어 처리 도구 설정"""
        try:
            if self.terms_manager and hasattr(self.terms_manager, 'search_index'):
                print("📚 중의학 전문용어를 jieba에 추가 중...")
                added_count = 0

                for term in self.terms_manager.search_index.keys():
                    if len(term) >= 2 and self._is_chinese_term(term):
                        jieba.add_word(term)
                        added_count += 1

                print(f"✅ {added_count}개 전문용어 추가 완료")
        except Exception as e:
            print(f"⚠️ 표준용어집 초기화 실패: {e}")
            self._setup_basic_jieba_terms()

    def _is_chinese_term(self, term: str) -> bool:
        """한자 용어인지 확인"""
        return all('\u4e00' <= char <= '\u9fff' for char in term)

    def _setup_basic_jieba_terms(self):
        """기본 중의학 용어를 jieba에 추가"""
        basic_terms = [
            "人參", "當歸", "川芎", "白芍", "熟地黃", "生地黃",
            "驚悸", "健忘", "癲癇", "眩暈", "失眠", "虛勞",
            "四物湯", "六君子湯", "補中益氣湯", "犀角地黃湯",
            "陰陽", "五行", "臟腑", "氣血", "經絡", "精氣神",
            "血虛", "氣虛", "陰虛", "陽虛", "脾胃", "肝腎"
        ]
        for term in basic_terms:
            jieba.add_word(term)
        print(f"✅ {len(basic_terms)}개 기본 중의학 용어 추가 완료")

    def load_basic_tcm_terms(self) -> Dict[str, List[str]]:
        """기본 중의학 전문 용어 사전 로드"""
        return {
            "symptoms": ["驚悸", "健忘", "癲癇", "眩暈", "失眠", "虛勞", "血虛", "氣虛"],
            "herbs": ["人參", "當歸", "川芎", "白芍", "熟地黃", "生地黃", "黃芪", "茯苓"],
            "prescriptions": ["四物湯", "六君子湯", "補中益氣湯", "犀角地黃湯", "當歸補血湯"],
            "theories": ["陰陽", "五行", "臟腑", "氣血", "經絡", "精氣神", "君臣佐使"]
        }

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
        """동의보감 문서 구조 파싱"""
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

                # 청킹 로직
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
        """처방 전용 청크 생성"""
        full_content = f"處方: {prescription_name}\n\n" + \
            '\n'.join(prescription_content)
        normalized_text, simplified_text = self.normalize_text(full_content)
        keywords = self.extract_prescription_keywords(
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

    def extract_prescription_keywords(self, prescription_name: str, content: List[str]) -> List[str]:
        """처방 전용 키워드 추출"""
        keywords = [prescription_name]
        full_text = ' '.join(content)

        # 약재명 패턴 매칭
        herb_patterns = [
            r'([一-龯]{2,4})\s*(?:各|每|用|取)',
            r'([一-龯]{2,4})\s*[一二三四五六七八九十百千만]\s*[錢兩分斤]',
        ]

        for pattern in herb_patterns:
            matches = re.findall(pattern, full_text)
            keywords.extend(matches)

        # 증상 키워드 추출
        for term_list in self.tcm_terms.values():
            for term in term_list:
                if term in full_text:
                    keywords.append(term)

        return list(set(keywords))[:15]

    def finalize_chunk(self, chunk: Dict, context: Dict) -> Dict:
        """청크 최종화"""
        content_text = ' '.join(chunk['content'])
        normalized_text, simplified_text = self.normalize_text(content_text)
        keywords = self.extract_keywords(normalized_text)

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

    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        keywords = []

        # 중의학 전문용어 매칭
        for category, terms in self.tcm_terms.items():
            for term in terms:
                if term in text:
                    keywords.append(term)

        # jieba 분할 결과 추가
        words = jieba.lcut(text)
        for word in words:
            if len(word) >= 2 and word not in keywords:
                keywords.append(word)

        return list(set(keywords))[:10]
