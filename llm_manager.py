#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 관리자 - OpenAI 전용 버전 (gpt-4o-mini)
dongui_rag_system.py에서 import하여 사용하는 독립 모듈
"""

import os
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import BaseMessage, SystemMessage, HumanMessage
    import tiktoken
except ImportError as e:
    print(f"필수 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치해주세요:")
    print("pip install langchain-community openai python-dotenv tiktoken")
    raise


class LLMManager:
    """OpenAI LLM 연결 및 관리 클래스 (gpt-4o-mini 전용)"""

    def __init__(self):
        """OpenAI LLM 관리자 초기화"""
        # 환경 변수 로드
        load_dotenv()

        # 모델 설정
        self.model_config = {
            'model_name': 'gpt-4o-mini',
            'max_context_tokens': 128000,
            'max_response_tokens': 4000,
            'safe_context_tokens': 120000,  # 안전 여유분 확보
            'optimal_k': 75,  # 최적 K값
            'timeout': 60
        }

        # 토크나이저 초기화
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # OpenAI 초기화
        self.llm = None
        self._available = False
        self._setup_openai()

    def _setup_openai(self):
        """OpenAI GPT-4o-mini 설정"""
        try:
            # API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                print("💡 .env 파일을 생성하거나 환경변수를 설정해주세요:")
                print("   OPENAI_API_KEY=your-api-key-here")
                self.llm = None
                return False

            # GPT-4o-mini 초기화
            self.llm = ChatOpenAI(
                model=self.model_config['model_name'],
                temperature=0,
                timeout=self.model_config['timeout'],
                max_tokens=self.model_config['max_response_tokens'],
                openai_api_key=api_key
            )

            # 연결 테스트
            print("🔗 OpenAI GPT-4o-mini 연결 테스트 중...")
            test_response = self.llm.invoke("안녕하세요")
            print("✅ OpenAI GPT-4o-mini 연결 성공!")
            print(f"📊 컨텍스트 길이: {self.model_config['max_context_tokens']:,} 토큰")
            print(f"🎯 최적 K값: {self.model_config['optimal_k']}개")
            self._available = True
            return True

        except Exception as e:
            print(f"❌ OpenAI 연결 실패: {e}")
            print("API 키를 확인하고 인터넷 연결을 확인해주세요.")
            self.llm = None
            self._available = False
            return False

    def is_available(self) -> bool:
        """LLM 사용 가능 여부 확인"""
        return self._available and self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        if not self.is_available():
            return {
                'type': 'none',
                'name': 'unavailable',
                'display_name': '사용 불가',
                'max_context_tokens': 0,
                'safe_context_tokens': 0,
                'optimal_k': 0,
                'is_connected': False
            }

        return {
            'type': 'openai',
            'name': self.model_config['model_name'],
            'display_name': 'OpenAI GPT-4o-mini',
            'max_context_tokens': self.model_config['max_context_tokens'],
            'safe_context_tokens': self.model_config['safe_context_tokens'],
            'optimal_k': self.model_config['optimal_k'],
            'is_connected': True
        }

    def get_optimal_k_value(self) -> int:
        """최적 K값 반환"""
        return self.model_config['optimal_k']

    def get_safe_context_tokens(self) -> int:
        """안전한 컨텍스트 토큰 수 반환"""
        return self.model_config['safe_context_tokens']

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        if not self.tokenizer:
            # 대략적인 추정 (한글/한자 고려)
            return len(text) // 2

        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 2

    def optimize_context_for_model(self, context_parts: List[str]) -> List[str]:
        """GPT-4o-mini에 맞게 컨텍스트 최적화"""
        if not context_parts:
            return []

        safe_tokens = self.get_safe_context_tokens()
        optimized_parts = []
        total_tokens = 0

        # 예약 토큰 (시스템 프롬프트 + 사용자 질문 + 응답용)
        reserved_tokens = 8000
        available_tokens = safe_tokens - reserved_tokens

        for part in context_parts:
            part_tokens = self.count_tokens(part)

            if total_tokens + part_tokens <= available_tokens:
                optimized_parts.append(part)
                total_tokens += part_tokens
            else:
                # 남은 토큰으로 부분 포함 시도
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 500:  # 최소 토큰 수 확보
                    truncated = self._truncate_text_safely(
                        part, remaining_tokens)
                    if truncated:
                        optimized_parts.append(truncated)
                break

        print(
            f"📊 컨텍스트 최적화: {len(context_parts)}개 → {len(optimized_parts)}개 문서")
        print(f"🔢 총 토큰 수: 약 {total_tokens:,}개 (제한: {available_tokens:,}개)")

        return optimized_parts

    def _truncate_text_safely(self, text: str, max_tokens: int) -> str:
        """텍스트를 안전하게 자르기"""
        if self.count_tokens(text) <= max_tokens:
            return text

        # 문장 단위로 자르기 시도
        sentences = text.split('.')
        truncated = ""

        for sentence in sentences:
            test_text = truncated + sentence + "."
            if self.count_tokens(test_text) <= max_tokens:
                truncated = test_text
            else:
                break

        if not truncated:
            # 문장 단위로도 안되면 문자 단위로
            estimated_chars = max_tokens * 2  # 대략적인 추정
            truncated = text[:estimated_chars] + "..."

        return truncated

    def generate_response(self, messages: List[BaseMessage]) -> str:
        """LLM으로부터 응답 생성"""
        if not self.is_available():
            return "OpenAI GPT-4o-mini가 연결되지 않아 답변을 생성할 수 없습니다. API 키를 확인해주세요."

        try:
            # 메시지 토큰 수 확인
            total_tokens = sum(self.count_tokens(msg.content)
                               for msg in messages)
            print(f"🔢 요청 토큰 수: 약 {total_tokens:,}개")

            if total_tokens > self.model_config['safe_context_tokens']:
                print("⚠️ 토큰 수가 많습니다. 응답이 잘릴 수 있습니다.")

            # OpenAI API 호출
            response = self.llm(messages)

            if not response or not response.content:
                return "OpenAI로부터 빈 응답을 받았습니다."

            return response.content

        except Exception as e:
            error_msg = str(e).lower()

            if 'rate limit' in error_msg:
                return "OpenAI API 요청 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
            elif 'api key' in error_msg or 'authentication' in error_msg:
                return "OpenAI API 키가 유효하지 않습니다. 설정을 확인해주세요."
            elif 'context length' in error_msg or 'token' in error_msg:
                return "요청한 내용이 너무 깁니다. 검색 결과 수를 줄여서 다시 시도해주세요."
            else:
                return f"OpenAI API 오류가 발생했습니다: {e}\n\n검색된 원문을 직접 확인해주세요."

    def print_model_status(self):
        """현재 모델 상태 출력"""
        model_info = self.get_model_info()
        status = "✅ 연결됨" if model_info['is_connected'] else "❌ 연결 안됨"

        print(f"🤖 현재 모델: {model_info['display_name']} ({status})")
        print(f"📊 최대 컨텍스트: {model_info['max_context_tokens']:,} 토큰")
        print(f"🎯 권장 K값: {model_info['optimal_k']}개")


# 편의 함수들
def create_llm_manager() -> LLMManager:
    """LLM 관리자 생성 편의 함수"""
    return LLMManager()


def setup_openai_api_key():
    """OpenAI API 키 설정 도우미"""
    print("\n🔑 OpenAI API 키 설정이 필요합니다.")
    print("1. https://platform.openai.com/api-keys 에서 API 키를 발급받으세요.")
    print("2. 다음 중 하나의 방법으로 설정하세요:")
    print()
    print("방법 1: .env 파일 생성 (권장)")
    print("   프로젝트 폴더에 .env 파일을 만들고 다음 내용을 추가:")
    print("   OPENAI_API_KEY=your-api-key-here")
    print()
    print("방법 2: 환경변수 설정")
    print("   터미널에서 다음 명령어 실행:")
    print("   export OPENAI_API_KEY=your-api-key-here")
    print()

    api_key = input("API 키를 직접 입력하시겠습니까? (Enter로 건너뛰기): ").strip()

    if api_key:
        # 임시로 환경변수에 설정
        os.environ['OPENAI_API_KEY'] = api_key
        print("✅ API 키가 임시로 설정되었습니다.")
        return True

    return False


def test_llm_connection() -> bool:
    """LLM 연결 테스트"""
    manager = LLMManager()
    return manager.is_available()


def main():
    """테스트용 메인 함수"""
    print("🧪 OpenAI LLM 관리자 테스트")

    llm_manager = LLMManager()

    if not llm_manager.is_available():
        if setup_openai_api_key():
            llm_manager = LLMManager()

        if not llm_manager.is_available():
            print("❌ OpenAI 설정을 완료한 후 다시 시도해주세요.")
            return

    # 모델 정보 출력
    llm_manager.print_model_status()

    # 간단한 테스트
    try:
        test_messages = [HumanMessage(content="안녕하세요! 간단한 인사말로 답해주세요.")]
        response = llm_manager.generate_response(test_messages)
        print(f"🤖 테스트 응답: {response}")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    main()
