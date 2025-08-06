#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시 관리 모듈 - cache_manager.py
시스템 데이터 캐싱과 로딩을 담당
"""

import pickle
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class CacheManager:
    def __init__(self, cache_path: str = "/Users/radi/Projects/langchainDATA/RAWDATA/DYBG/cache"):
        """캐시 관리자 초기화"""
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def save_cache(self, data_hash: str, chunks: List[Dict], embeddings: np.ndarray, faiss_index):
        """캐시 저장"""
        print("💾 캐시 저장 중...")

        cache_data = {
            'data_hash': data_hash,
            'chunks': chunks,
            'timestamp': datetime.now().isoformat()
        }

        # 청크 데이터 저장
        chunks_file = self.cache_path / 'chunks.pkl'
        with open(chunks_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # 임베딩 저장
        embeddings_file = self.cache_path / 'embeddings.npy'
        np.save(embeddings_file, embeddings)

        # FAISS 인덱스 저장
        faiss_file = self.cache_path / 'faiss_index.index'
        faiss.write_index(faiss_index, str(faiss_file))

        print("✅ 캐시 저장 완료")

    def load_cache(self, current_data_hash: str) -> Tuple[bool, Optional[Dict]]:
        """캐시 로드"""
        chunks_file = self.cache_path / 'chunks.pkl'
        embeddings_file = self.cache_path / 'embeddings.npy'
        faiss_file = self.cache_path / 'faiss_index.index'

        if not all([chunks_file.exists(), embeddings_file.exists(), faiss_file.exists()]):
            print("📝 캐시 파일이 없습니다. 새로 생성합니다.")
            return False, None

        try:
            print("🔄 캐시에서 데이터 로딩 중...")

            # 청크 데이터 로드
            with open(chunks_file, 'rb') as f:
                cache_data = pickle.load(f)

            # 데이터 해시 비교
            if cache_data['data_hash'] != current_data_hash:
                print("📝 원본 데이터가 변경되었습니다. 새로 생성합니다.")
                return False, None

            # 나머지 데이터 로드
            embeddings = np.load(embeddings_file)
            faiss_index = faiss.read_index(str(faiss_file))

            result = {
                'data_hash': cache_data['data_hash'],
                'chunks': cache_data['chunks'],
                'embeddings': embeddings,
                'faiss_index': faiss_index,
                'timestamp': cache_data['timestamp']
            }

            print(f"✅ 캐시 로드 완료! (생성: {cache_data['timestamp']})")
            print(
                f"📊 로드된 데이터: {len(cache_data['chunks'])}개 청크, {embeddings.shape} 임베딩")
            return True, result

        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            print("📝 새로 생성합니다.")
            return False, None

    def clear_cache(self):
        """캐시 삭제"""
        try:
            cache_files = [
                self.cache_path / 'chunks.pkl',
                self.cache_path / 'embeddings.npy',
                self.cache_path / 'faiss_index.index'
            ]

            for file in cache_files:
                if file.exists():
                    file.unlink()

            print("🗑️ 캐시가 삭제되었습니다.")
        except Exception as e:
            print(f"⚠️ 캐시 삭제 실패: {e}")

    def get_cache_info(self) -> Dict:
        """캐시 정보 조회"""
        chunks_file = self.cache_path / 'chunks.pkl'
        embeddings_file = self.cache_path / 'embeddings.npy'
        faiss_file = self.cache_path / 'faiss_index.index'

        cache_info = {
            'chunks_exists': chunks_file.exists(),
            'embeddings_exists': embeddings_file.exists(),
            'faiss_exists': faiss_file.exists(),
            'cache_complete': False,
            'size_info': {}
        }

        if all([chunks_file.exists(), embeddings_file.exists(), faiss_file.exists()]):
            cache_info['cache_complete'] = True

            try:
                # 파일 크기 정보
                cache_info['size_info'] = {
                    'chunks_size_mb': chunks_file.stat().st_size / (1024 * 1024),
                    'embeddings_size_mb': embeddings_file.stat().st_size / (1024 * 1024),
                    'faiss_size_mb': faiss_file.stat().st_size / (1024 * 1024)
                }

                # 청크 개수 정보
                with open(chunks_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cache_info['chunks_count'] = len(cache_data['chunks'])
                    cache_info['timestamp'] = cache_data['timestamp']
                    cache_info['data_hash'] = cache_data['data_hash']

            except Exception as e:
                cache_info['error'] = str(e)

        return cache_info
