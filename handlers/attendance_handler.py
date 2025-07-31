from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document  
from langchain_community.vectorstores import Chroma  
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import os 
from dotenv import load_dotenv

load_dotenv()

# 현재 파일 기준으로 경로 설정
current_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Google API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Gemini 모델 초기화 (main_chat_two.py와 동일한 설정)
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key,
    temperature=0
)

# vector_store_two.py에서 사용한 것과 동일한 임베딩 모델 사용
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# 벡터DB 경로 설정 (vector_store_two.py와 동일한 경로)
vector_db_path = os.path.join(base_dir, "my_rag_db")
collection_name = "admin_docs"

# 기존 벡터 저장소 로드
vector_db = Chroma(
    persist_directory=vector_db_path,
    embedding_function=embeddings,
    collection_name=collection_name
)

def retrieve_chunks(query: str, k: int = 5) -> list[Document]:
    """
    사용자 질의에 대해 벡터 저장소에서 관련 문서 조각들을 검색합니다.
    """
    # 출석 도메인의 다양한 용어들 매핑
    attendance_terms = {
        "QR": "QR체크 QR코드 출석체크 출결확인",
        "인정": "출석인정 출결인정",
        "출석": "출석체크 출결 출석인정 입실 퇴실",
        "출결": "출석체크 출결확인 출석인정 출결정정",
        "외출": "외출신청 단순외출 외출처리",
        "정정": "출결정정 출석정정 정정신청",
        "지각": "지각처리 입실지각 늦음",
        "조퇴": "조퇴처리 퇴실조퇴 일찍",
        "결석": "결석처리 불참",
        "HRD": "HRD오류 시스템오류 서버오류",
        "스크린샷": "스샷 캡처 화면캡처",
        "카메라": "카메라ON 웹캠",
        "원격": "원격수업 온라인수업",
        "증빙": "증빙서류 증명서류 서류제출",
        "신청": "정정신청 출결신청 요청",
        "미체크": "QR미체크 체크안됨 인식안됨"
    }
    
    # 검색 쿼리 확장
    enhanced_query = query
    for term, expansion in attendance_terms.items():
        if term in query:
            enhanced_query += f" {expansion}"

    # 벡터DB에서 유사도 검색 수행
    return vector_db.similarity_search(enhanced_query, k=k)

def generate_answer(query: str, chunks: list[Document]) -> str:
    """
    검색된 문서 조각들을 바탕으로 사용자 질의에 대한 답변을 생성합니다.
    """
    # 문서 조각들을 하나의 텍스트로 결합
    docs_text = "\n\n".join(
        f"[{i+1}] (출처: {d.metadata.get('source', '알 수 없음')})\n{d.page_content}"
        for i, d in enumerate(chunks)
    )
    
    # 간결한 답변을 위한 프롬프트 템플릿
    system_prompt = """당신은 패스트캠퍼스 출석 관리 챗봇입니다.
문서 내용을 바탕으로 간결하고 핵심적인 답변만 제공하세요.

**답변 규칙:**
1. 3-4문장 이내로 간결하게 답변합니다.
2. 가장 중요한 정보만 선별해서 전달합니다.
3. 조건이나 기한이 있으면 반드시 포함합니다.
4. 불가능한 경우 "정정 불가" 또는 "대상 아님"으로 명확히 답변합니다.
5. 존댓말을 사용합니다.

**핵심 규칙:**
- 단순 실수 QR 미체크 = 정정 불가
- HRD 오류 + 스크린샷 참여 = 정정 가능
- 신청 기한: 영업일 기준 다음날 16시까지

관련 문서:
{docs_text}

사용자 질문: {query}

간결하고 핵심적인 답변을 제공해주세요."""

    prompt_template = PromptTemplate(
        template=system_prompt,
        input_variables=["docs_text", "query"]
    )
    
    # LLMChain을 사용하여 답변 생성
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    try:
        response = chain.run(docs_text=docs_text, query=query)
        return response.strip()
    except Exception as e:
        print(f"[❌ Gemini LLM 호출 오류]: {e}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."

def answer(query: str, top_k: int = 5, student_id: str = None, student_info: dict = None) -> str:
    """
    출석 관련 질의에 대한 RAG 기반 답변을 제공하는 메인 함수입니다.
    
    Args:
        query (str): 사용자 질의
        top_k (int): 검색할 문서 개수 (기본값: 5)
        student_id (str): 학생 ID (선택사항)
        student_info (dict): 학생 정보 (선택사항)
    
    Returns:
        str: 생성된 답변
    """
    try:
        # 입력 검증
        if not query or not query.strip():
            return "출석 관련 질문을 입력해주세요."
        
        # ① 검색 단계: 관련 문서 조각 검색
        chunks = retrieve_chunks(query.strip(), k=top_k)
        
        # 검색된 문서가 없는 경우 처리
        if not chunks:
            return "죄송합니다. 출석 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
        
        # ② 답변 생성 단계: 검색된 문서를 바탕으로 답변 생성
        return generate_answer(query.strip(), chunks)
        
    except Exception as e:
        print(f"[❌ attendance_handler 전체 처리 오류]: {e}")
        return "출석 관련 문의 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

if __name__ == "__main__":
    # 테스트 코드
    print("=== 출석 핸들러 테스트 ===")
    
    # 출결정정 관련 테스트 질의
    test_queries = [
        "출석체크는 어떻게 해야 하나요?",
        "QR 코드로 출석 인증이 안 됐어요. 어떻게 정정하나요?",
        "외출할 때 신청해야 하나요?",
        "출결정정 신청은 어떻게 하나요?",
        "HRD 오류로 QR 체크가 안됐는데 정정 가능한가요?",
        "출결정정 신청 기한이 언제까지인가요?",
        "스크린샷 참여를 안했는데 출결정정 가능한가요?",
        "단순 실수로 QR 체크를 못했는데 정정 가능한가요?",
        "원격수업에서 카메라를 안켰는데 어떻게 되나요?",
        "지각이나 조퇴는 어떻게 처리되나요?",
        "출석대장 작성할 때 주의사항이 있나요?",
        "출결정정 증빙서류는 언제까지 제출해야 하나요?"
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        print(f"답변: {answer(query)}")
        print("-" * 50)