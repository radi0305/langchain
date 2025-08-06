#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의보감 검색 엔진 모듈 - search_engine.py
임베딩, 인덱싱, 검색 로직을 담당
"""

import numpy as np
import faiss
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")


class SearchEngine:
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """검색 엔진 초기화"""
        print("🔄 임베딩 모델 로딩 중...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("✅ 임베딩 모델 로딩 완료")

        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.terms_manager = None

    def set_terms_manager(self, terms_manager):
        """표준용어집 관리자 설정"""
        self.terms_manager = terms_manager

    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """임베딩 생성"""
        print("🔄 임베딩 생성 중...")

        texts = []
        for chunk in chunks:
            title_parts = []
            if chunk['metadata'].get('BB'):
                title_parts.append(chunk['metadata']['BB'])
            if chunk['metadata'].get('CC'):
                title_parts.append(chunk['metadata']['CC'])

            title = ' '.join(title_parts)
            combined_text = f"{title} {chunk['content']}" if title else chunk['content']
            texts.append(combined_text)

        # 배치 처리로 임베딩 생성
        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, show_progress_bar=True, batch_size=batch_size)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        print(f"✅ 임베딩 생성 완료: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray):
        """FAISS 인덱스 구축"""
        print("🔍 FAISS 인덱스 구축 중...")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # 임베딩 정규화 (코사인 유사도)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))

        print(f"✅ FAISS 인덱스 구축 완료: {index.ntotal}개 벡터")
        return index

    def setup(self, chunks: List[Dict], embeddings: np.ndarray = None):
        """검색 엔진 설정"""
        self.chunks = chunks

        if embeddings is None:
            self.embeddings = self.create_embeddings(chunks)
        else:
            self.embeddings = embeddings

        self.faiss_index = self.build_faiss_index(self.embeddings)

    def search(self, query: str, k: int = 75) -> List[Dict]:
        """메인 검색 함수"""
        if self.faiss_index is None:
            raise ValueError("검색 엔진이 설정되지 않았습니다.")

        print(f"🔍 '{query}' 검색 전략 수립 중...")

        # 1단계: 대량 후보 수집
        all_candidates = self._collect_comprehensive_candidates(query, k)

        # 2단계: 클러스터링 분석
        clustered_candidates = self._cluster_by_content_type(
            query, all_candidates)

        # 3단계: 대표 선정
        final_results = self._select_representatives(clustered_candidates, k)

        print(f"✅ 최종 {len(final_results)}개 대표 결과 선정")
        return final_results

    def _semantic_search(self, query: str, k: int) -> List[Dict]:
        """기본 시맨틱 검색"""
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'semantic_score': float(score),
                    'chunk_index': idx
                })

        return results

    def _enhanced_keyword_search(self, query: str, k: int) -> List[Dict]:
        """강화된 키워드 검색"""
        keyword_results = []

        for i, chunk in enumerate(self.chunks):
            content = chunk['content']
            score = 0.0

            # 완전 매칭
            if query in content:
                score += 5.0

            # 부분 매칭
            for char in query:
                if char in content:
                    score += 0.2

            # 처방명 매칭
            if chunk['metadata'].get('prescription_name'):
                prescription_name = chunk['metadata']['prescription_name']
                if query in prescription_name or prescription_name in query:
                    score += 8.0

            # 키워드 매칭
            keywords = chunk['metadata'].get('keywords', [])
            for keyword in keywords:
                if query in keyword or keyword in query:
                    score += 1.0

            # 위치 가중치
            if query in content:
                position = content.find(query)
                position_weight = max(0, 1.0 - (position / len(content)))
                score += position_weight * 2.0

            if score > 1.0:
                keyword_results.append({
                    'content': content,
                    'metadata': chunk['metadata'],
                    'semantic_score': score,
                    'chunk_index': i
                })

        keyword_results.sort(key=lambda x: x['semantic_score'], reverse=True)
        return keyword_results[:k]

    def _context_based_search(self, query: str, k: int) -> List[Dict]:
        """컨텍스트 기반 검색"""
        context_results = []
        query_context = self._analyze_query_context(query)

        for i, chunk in enumerate(self.chunks):
            content = chunk['content']
            metadata = chunk['metadata']
            score = 0.0

            # 컨텍스트 매칭
            if query_context['is_symptom'] and '證' in content:
                score += 2.0
            if query_context['is_prescription'] and metadata.get('type') == 'prescription':
                score += 3.0
            if query_context['is_theory'] and metadata.get('BB'):
                score += 1.5

            # 관련 용어 매칭
            related_terms = query_context.get('related_terms', [])
            for term in related_terms:
                if term in content:
                    score += 0.8

            # 카테고리 매칭
            if query_context['category'] and query_context['category'] == metadata.get('BB'):
                score += 2.0

            if score > 0.5:
                context_results.append({
                    'content': content,
                    'metadata': metadata,
                    'semantic_score': score,
                    'chunk_index': i
                })

        context_results.sort(key=lambda x: x['semantic_score'], reverse=True)
        return context_results[:k]

    def expand_query(self, query: str) -> List[str]:
        """쿼리 확장"""
        try:
            if self.terms_manager:
                # 표준용어집 기반 확장
                standard_expansions = self.terms_manager.expand_query(
                    query, max_expansions=10)
                query_parts = self.terms_manager.split_query_intelligently(
                    query)
                for part in query_parts:
                    if part not in standard_expansions:
                        standard_expansions.append(part)
                return standard_expansions[:12]
        except Exception as e:
            print(f"⚠️ 표준용어집 기반 확장 실패: {e}")

        # 폴백: 기본 확장
        return [query]

    def _collect_comprehensive_candidates(self, query: str, k: int) -> List[Dict]:
        """포괄적 후보 수집"""
        all_candidates = []

        # 직접 매칭
        direct_results = self._semantic_search(query, k * 8)
        for r in direct_results:
            r['search_strategy'] = 'direct'
            r['relevance_boost'] = 1.0
        all_candidates.extend(direct_results)

        # 확장 검색
        expanded_queries = self.expand_query(query)
        for i, exp_query in enumerate(expanded_queries[1:]):
            exp_results = self._semantic_search(exp_query, k * 3)
            for r in exp_results:
                r['search_strategy'] = 'expanded'
                r['expanded_query'] = exp_query
                r['relevance_boost'] = 0.9 - (i * 0.02)
            all_candidates.extend(exp_results)

        # 키워드 검색
        keyword_results = self._enhanced_keyword_search(query, k * 6)
        for r in keyword_results:
            r['search_strategy'] = 'keyword'
            r['relevance_boost'] = 0.95
        all_candidates.extend(keyword_results)

        # 컨텍스트 검색
        context_results = self._context_based_search(query, k * 4)
        for r in context_results:
            r['search_strategy'] = 'context'
            r['relevance_boost'] = 0.85
        all_candidates.extend(context_results)

        print(f"📊 총 {len(all_candidates)}개 후보 수집")

        # 고급 스코어링
        scored_results = self._enhanced_scoring(query, all_candidates)
        return scored_results[:k * 20]

    def _analyze_query_context(self, query: str) -> Dict:
        """쿼리 컨텍스트 분석"""
        context = {
            'is_symptom': False,
            'is_prescription': False,
            'is_theory': False,
            'category': None,
            'related_terms': []
        }

        # 증상/처방/이론 감지
        symptom_indicators = ['虛', '證', '病', '症', '痛', '熱', '寒', '濕', '燥']
        if any(indicator in query for indicator in symptom_indicators):
            context['is_symptom'] = True

        prescription_indicators = ['湯', '散', '丸', '膏', '處方']
        if any(indicator in query for indicator in prescription_indicators):
            context['is_prescription'] = True

        theory_indicators = ['陰陽', '五行', '氣血', '精氣神']
        if any(indicator in query for indicator in theory_indicators):
            context['is_theory'] = True

        # 표준용어집 기반 관련 용어 추출
        try:
            if self.terms_manager:
                context['related_terms'] = self.terms_manager.get_related_terms(
                    query)
        except Exception:
            context['related_terms'] = self._extract_basic_related_terms(query)

        return context

    def _extract_basic_related_terms(self, query: str) -> List[str]:
        """기본적인 관련 용어 추출"""
        basic_relations = []

        if '虛' in query:
            basic_relations.extend(['補', '益', '養', '調', '溫'])
        if '血' in query:
            basic_relations.extend(['氣', '陰', '陽', '補血', '養血'])
        if '氣' in query:
            basic_relations.extend(['血', '補氣', '益氣', '理氣'])
        if '湯' in query:
            basic_relations.extend(['處方', '方劑', '治療'])

        return basic_relations

    def _cluster_by_content_type(self, query: str, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """내용 타입별 클러스터링"""
        clusters = {
            'direct_match': [],      # 직접 매칭
            'prescription': [],      # 처방 관련
            'symptom_theory': [],    # 증상/이론
            'related_concept': [],   # 관련 개념
            'treatment_method': [],  # 치료법
            'differential': []       # 감별진단
        }

        for candidate in candidates:
            content = candidate['content']
            metadata = candidate['metadata']

            # 클러스터 분류
            if query in content and candidate.get('semantic_score', 0) > 4.0:
                clusters['direct_match'].append(candidate)
            elif metadata.get('type') == 'prescription':
                clusters['prescription'].append(candidate)
            elif any(keyword in content for keyword in ['證', '病', '症', '痛']):
                clusters['symptom_theory'].append(candidate)
            elif any(keyword in content for keyword in ['治', '療', '主', '用']):
                clusters['treatment_method'].append(candidate)
            elif self._is_related_concept(query, content):
                clusters['related_concept'].append(candidate)
            else:
                clusters['differential'].append(candidate)

        # 각 클러스터 내 정렬
        for cluster_name, cluster_items in clusters.items():
            clusters[cluster_name] = sorted(cluster_items,
                                            key=lambda x: x.get(
                                                'final_score', x.get('semantic_score', 0)),
                                            reverse=True)

        print(f"📋 클러스터 분포: 직접매칭={len(clusters['direct_match'])}, "
              f"처방={len(clusters['prescription'])}, 증상이론={len(clusters['symptom_theory'])}, "
              f"치료법={len(clusters['treatment_method'])}, 관련개념={len(clusters['related_concept'])}, "
              f"기타={len(clusters['differential'])}")

        return clusters

    def _is_related_concept(self, query: str, content: str) -> bool:
        """관련 개념 판단"""
        try:
            if self.terms_manager:
                related_terms = self.terms_manager.get_related_terms(query)
                return any(term in content for term in related_terms)
        except Exception:
            pass

        # 기본적인 관련성 판단
        if any(char in query for char in ['虛', '氣', '血', '陰', '陽']):
            related_patterns = ['虛', '氣', '血', '陰', '陽', '補', '益', '養']
            return any(pattern in content for pattern in related_patterns)

        return False

    def _enhanced_scoring(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """강화된 스코어링 시스템"""
        for candidate in candidates:
            final_score = 0.0

            # 기본 시맨틱 점수
            semantic_score = candidate.get('semantic_score', 0.0)
            final_score += semantic_score * 0.3

            # 검색 전략별 가중치
            strategy_weight = {
                'direct': 1.0,
                'expanded': 0.8,
                'keyword': 0.9,
                'context': 0.7
            }
            strategy = candidate.get('search_strategy', 'direct')
            final_score *= strategy_weight.get(strategy, 0.5)

            # 관련성 부스트
            relevance_boost = candidate.get('relevance_boost', 1.0)
            final_score *= relevance_boost

            # 키워드 매칭 보너스
            keyword_bonus = self._calculate_keyword_bonus(query, candidate)
            final_score += keyword_bonus * 0.4

            # 컨텍스트 품질 보너스
            context_bonus = self._calculate_context_bonus(candidate)
            final_score += context_bonus * 0.2

            # 타입 보너스
            type_bonus = self._calculate_type_bonus(query, candidate)
            final_score += type_bonus * 0.1

            candidate['final_score'] = final_score

        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

    def _calculate_keyword_bonus(self, query: str, candidate: Dict) -> float:
        """키워드 매칭 보너스 계산"""
        content = candidate['content'].lower()
        query_lower = query.lower()
        bonus = 0.0

        # 완전 매칭
        if query_lower in content:
            bonus += 2.0

        # 처방명 매칭
        prescription_name = candidate['metadata'].get(
            'prescription_name', '').lower()
        if prescription_name and (query_lower in prescription_name or prescription_name in query_lower):
            bonus += 3.0

        # 글자별 매칭
        matched_chars = sum(1 for char in query_lower if char in content)
        char_ratio = matched_chars / len(query_lower) if query_lower else 0
        bonus += char_ratio * 0.5

        # 키워드 리스트 매칭
        keywords = candidate['metadata'].get('keywords', [])
        for keyword in keywords:
            if keyword.lower() in query_lower or query_lower in keyword.lower():
                bonus += 0.3

        return min(bonus, 3.0)

    def _calculate_context_bonus(self, candidate: Dict) -> float:
        """컨텍스트 품질 보너스"""
        bonus = 0.0
        metadata = candidate['metadata']

        # 내용 길이 적정성
        content_len = len(candidate['content'])
        if 200 <= content_len <= 1000:
            bonus += 1.0
        elif 100 <= content_len <= 1500:
            bonus += 0.5

        # 메타데이터 완성도
        if metadata.get('BB'):
            bonus += 0.3
        if metadata.get('CC'):
            bonus += 0.3
        if metadata.get('prescription_name'):
            bonus += 0.4

        return bonus

    def _calculate_type_bonus(self, query: str, candidate: Dict) -> float:
        """문서 타입 보너스"""
        content_type = candidate['metadata'].get('type', 'general')

        # 처방 관련 쿼리 감지
        prescription_keywords = ['湯', '散', '丸', '膏', '處方', '치료']
        is_prescription_query = any(
            kw in query for kw in prescription_keywords)

        if is_prescription_query and content_type == 'prescription':
            return 1.0
        elif content_type == 'prescription':
            return 0.5
        else:
            return 0.0

    def _select_representatives(self, clusters: Dict[str, List[Dict]], k: int) -> List[Dict]:
        """대표 선정 (상용 AI 모델용)"""

        # 상용 AI 모델용 할당 전략
        if k < 50:
            allocation_strategy = {
                'direct_match': 0.35,     # 35% - 직접 매칭
                'prescription': 0.25,     # 25% - 처방 정보
                'symptom_theory': 0.20,   # 20% - 증상/이론
                'treatment_method': 0.15,  # 15% - 치료법
                'related_concept': 0.05,  # 5% - 관련 개념
                'differential': 0.0       # 0% - 기타
            }
        elif k <= 70:
            allocation_strategy = {
                'direct_match': 0.30,     # 30% - 직접 매칭
                'prescription': 0.25,     # 25% - 처방 정보
                'symptom_theory': 0.20,   # 20% - 증상/이론
                'treatment_method': 0.15,  # 15% - 치료법
                'related_concept': 0.07,  # 7% - 관련 개념
                'differential': 0.03      # 3% - 감별진단
            }
        else:
            allocation_strategy = {
                'direct_match': 0.25,     # 25% - 직접 매칭
                'prescription': 0.25,     # 25% - 처방 정보
                'symptom_theory': 0.20,   # 20% - 증상/이론
                'treatment_method': 0.15,  # 15% - 치료법
                'related_concept': 0.10,  # 10% - 관련 개념
                'differential': 0.05      # 5% - 감별진단
            }

        # 실제 사용 가능한 후보 수 확인
        total_available = sum(len(cluster_items)
                              for cluster_items in clusters.values())
        actual_k = min(k, total_available)

        if actual_k < k:
            print(f"⚠️ 요청된 K값({k})보다 사용 가능한 후보가 적습니다({total_available}개)")
            print(f"📊 실제 선정 가능한 수: {actual_k}개")

        selected = []
        used_content_hashes = set()
        cluster_stats = {}

        # 1단계: 각 클러스터에서 할당된 비율만큼 선정
        for cluster_name, ratio in allocation_strategy.items():
            cluster_items = clusters.get(cluster_name, [])

            if not cluster_items:
                cluster_stats[cluster_name] = {
                    'target': 0, 'selected': 0, 'available': 0}
                continue

            target_count = max(1, int(actual_k * ratio)) if ratio > 0 else 0

            cluster_selected = 0
            for item in cluster_items:
                if cluster_selected >= target_count or len(selected) >= actual_k:
                    break

                content_hash = hash(item['content'])
                if content_hash in used_content_hashes:
                    continue

                selected.append(item)
                used_content_hashes.add(content_hash)
                cluster_selected += 1

            cluster_stats[cluster_name] = {
                'target': target_count,
                'selected': cluster_selected,
                'available': len(cluster_items)
            }

            if len(selected) >= actual_k:
                break

        # 2단계: 부족한 경우 최고 점수 항목들로 채우기
        if len(selected) < actual_k:
            print(f"🔄 {len(selected)}/{actual_k}개 선정됨. 추가 항목으로 보충합니다.")

            all_remaining = []
            for cluster_items in clusters.values():
                all_remaining.extend(cluster_items)

            all_remaining.sort(key=lambda x: x.get(
                'final_score', x.get('semantic_score', 0)), reverse=True)

            for item in all_remaining:
                if len(selected) >= actual_k:
                    break

                content_hash = hash(item['content'])
                if content_hash not in used_content_hashes:
                    selected.append(item)
                    used_content_hashes.add(content_hash)

        # 3단계: 대용량에서 다양성 보장 (K≥50)
        if actual_k >= 50:
            diversity_enhanced = self._ensure_diversity_for_large_k(
                selected, actual_k)
            selected = diversity_enhanced

        # 4단계: 최종 정렬 및 score 필드 설정
        for item in selected:
            if 'score' not in item:
                item['score'] = item.get(
                    'final_score', item.get('semantic_score', 0))

        final_selected = sorted(
            selected, key=lambda x: x['score'], reverse=True)[:actual_k]

        # 상세 통계 출력
        print(f"\n📊 클러스터별 선정 결과:")
        for cluster_name, stats in cluster_stats.items():
            print(f"   {cluster_name}: {stats['selected']}/{stats['target']}개 선정 "
                  f"(사용가능: {stats['available']}개)")

        print(f"\n🎯 최종 선정: {len(final_selected)}개 (요청: {k}개, 실제: {actual_k}개)")

        return final_selected

    def _ensure_diversity_for_large_k(self, selected: List[Dict], k: int) -> List[Dict]:
        """대용량 K값(50+)에서 다양성 보장"""

        if k < 50:
            return selected

        # 출처 파일별 분포 최적화
        max_per_source = max(5, k // 8)  # 최대 12.5% 또는 최소 5개
        # 대분류(BB)별 분포 제한
        max_per_bb = max(8, k // 6)  # 최대 16.7% 또는 최소 8개

        adjusted_selected = []
        source_counts = {}
        bb_counts = {}

        # 1차: 출처와 대분류 다양성을 고려한 선정
        for item in sorted(selected, key=lambda x: x.get('score', 0), reverse=True):
            source = item['metadata'].get('source_file', 'unknown')
            bb = item['metadata'].get('BB', 'unknown')

            source_count = source_counts.get(source, 0)
            bb_count = bb_counts.get(bb, 0)

            if source_count < max_per_source and bb_count < max_per_bb:
                adjusted_selected.append(item)
                source_counts[source] = source_count + 1
                bb_counts[bb] = bb_count + 1

            if len(adjusted_selected) >= k:
                break

        # 2차: 부족한 경우 제한을 완화하여 채우기
        if len(adjusted_selected) < k:
            remaining = [
                item for item in selected if item not in adjusted_selected]
            needed = k - len(adjusted_selected)

            remaining.sort(key=lambda x: x.get('score', 0), reverse=True)
            adjusted_selected.extend(remaining[:needed])

        # 통계 정보 출력
        final_source_dist = {}
        final_bb_dist = {}
        final_type_dist = {}

        for item in adjusted_selected:
            source = item['metadata'].get('source_file', 'unknown')
            bb = item['metadata'].get('BB', 'unknown')
            content_type = item['metadata'].get('type', 'general')

            final_source_dist[source] = final_source_dist.get(source, 0) + 1
            final_bb_dist[bb] = final_bb_dist.get(bb, 0) + 1
            final_type_dist[content_type] = final_type_dist.get(
                content_type, 0) + 1

        print(
            f"📚 출처 다양성: {len(final_source_dist)}개 파일 (최대 {max_per_source}개/파일)")
        print(f"📖 대분류 다양성: {len(final_bb_dist)}개 영역 (최대 {max_per_bb}개/영역)")
        print(f"🏷️ 내용 타입: {final_type_dist}")

        return adjusted_selected
