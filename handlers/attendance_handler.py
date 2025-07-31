from langchain_openai import ChatOpenAI
from langchain.schema import Document  
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma  

import os 
from dotenv import load_dotenv
import re

load_dotenv()

# API 키는 .env에서 자동 로드되므로 여기선 생략, temperature=0(결과 재현성 높게)로 생성
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.05)  

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 기존에 생성된 Chroma 벡터 DB 로드
vector_db = Chroma(
    persist_directory="./my_rag_db", 
    embedding_function=embeddings,
    collection_name="admin_docs"
)

# 1. 질문 전처리 및 확장 함수
def preprocess_query(query: str) -> str:
    """사용자 질문을 검색에 적합하게 전처리"""
    # 기본 정규화
    query = query.strip()
    
    # 구어체 -> 표준어 변환
    replacements = {
        '못 찍': '스캔 안됨',
        '어떡해': '어떻게 해야',
        '뭐야': '무엇인가요',
        '어케': '어떻게',
        '안돼': '안됩니다'
    }
    
    for old, new in replacements.items():
        query = query.replace(old, new)
    
    return query

def expand_query(query: str) -> str:
    """질문 의도에 따른 키워드 확장"""
    enhanced_query = query
    
    # QR 관련 질문 확장
    if any(word in query.lower() for word in ['qr', '큐알', '스캔', '찍', '코드']):
        enhanced_query += " QR코드 출석체크 출결확인 스캔 인식 오류 대안방법"
    
    # 출석 인정 관련
    if any(word in query.lower() for word in ['인정', '인정받', '승인']):
        enhanced_query += " 출석인정 출결인정 승인 인정기준"
    
    # 정정 신청 관련  
    if any(word in query.lower() for word in ['정정', '수정', '변경', '바꾸']):
        enhanced_query += " 정정신청 출석정정 수정 변경 절차"
    
    # 스크린샷 관련
    if any(word in query.lower() for word in ['스크린샷', '화면', '캡처', '증빙']):
        enhanced_query += " 스크린샷 화면캡처 증빙자료 제출 요건"
    
    # 지각/결석 관련
    if any(word in query.lower() for word in ['지각', '결석', '늦', '빠짐']):
        enhanced_query += " 지각 결석 늦음 조퇴 출석처리"
    
    # 문의/신청 관련
    if any(word in query.lower() for word in ['문의', '신청', '어디', '어떻게']):
        enhanced_query += " 문의 신청 절차 방법 담당자 연락처"
    
    return enhanced_query

# 2. 향상된 검색 함수
def retrieve_chunks(query: str, k: int = 5) -> list[Document]:
    """다단계 검색으로 정확도 향상"""
    
    # 1단계: 전처리된 질문으로 검색
    processed_query = preprocess_query(query)
    enhanced_query = expand_query(processed_query)
    
    print(f"🔍 원본 질문: {query}")
    print(f"🔍 확장된 질문: {enhanced_query}")
    
    # 2단계: 기본 유사도 검색
    primary_results = vector_db.similarity_search(enhanced_query, k=k*2)
    
    # 3단계: MMR(Maximum Marginal Relevance) 검색으로 다양성 확보
    try:
        mmr_results = vector_db.max_marginal_relevance_search(
            enhanced_query, 
            k=k, 
            fetch_k=k*2,
            lambda_mult=0.7  # 관련성과 다양성 균형
        )
    except:
        mmr_results = primary_results[:k]
    
    # 4단계: 결과 품질 평가 및 필터링
    filtered_results = []
    query_keywords = set(re.findall(r'\w+', query.lower()))
    
    for doc in mmr_results:
        content_lower = doc.page_content.lower()
        
        # 키워드 매칭 점수 계산
        matching_keywords = sum(1 for keyword in query_keywords if keyword in content_lower)
        
        # 최소 관련성 기준 (키워드가 하나도 없으면 제외)
        if matching_keywords > 0 or len(filtered_results) < 2:  # 최소 2개는 보장
            filtered_results.append(doc)
    
    return filtered_results[:k]

# 3. 향상된 답변 생성 함수
def generate_answer(query: str, chunks: list[Document]) -> str:
    """문맥을 고려한 정확한 답변 생성"""
    
    if not chunks:
        return "죄송합니다. 해당 질문과 관련된 문서를 찾을 수 없습니다. 다른 표현으로 질문해 주시거나 관리자에게 직접 문의해 주세요."
    
    # 문서 정보를 구조화하여 제공
    docs_text = ""
    for i, doc in enumerate(chunks):
        source_info = doc.metadata.get('source_file') or doc.metadata.get('source', '알 수 없음')
        keywords = doc.metadata.get('keywords', '')
        
        docs_text += f"[문서 {i+1}] (출처: {source_info})\n"
        if keywords:
            docs_text += f"관련키워드: {keywords}\n"
        docs_text += f"{doc.page_content}\n\n"
    
    # 향상된 시스템 프롬프트
    system_prompt = (
        "당신은 패스트캠퍼스 수강생을 위한 전문 행정 챗봇입니다.\n\n"
        "답변 원칙:\n"
        "1. 제공된 문서의 정확한 정보만을 기반으로 답변하세요\n"
        "2. 사용자 질문이 구어체나 불완전한 문장이어도 의도를 파악하여 친절하게 답변하세요\n"
        "3. 구체적인 절차나 방법이 있다면 단계별로 설명하세요\n"
        "4. 문서에 명시되지 않은 내용은 '문서에서 확인되지 않습니다'라고 명확히 답변하세요\n"
        "5. 연락처나 담당자 정보가 있다면 함께 제공하세요\n"
        "6. 답변은 친근하고 이해하기 쉽게 작성하세요"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"사용자 질문: \"{query}\"\n\n"
            f"관련 문서 내용:\n{docs_text}\n\n"
            "위 문서 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변해 주세요:"
        ))
    ]
    
    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
        
        # 답변 품질 검증
        if len(answer) < 10:
            return "죄송합니다. 적절한 답변을 생성하지 못했습니다. 질문을 다시 정리해서 문의해 주세요."
        
        return answer
    
    except Exception as e:
        print(f"⚠️ 답변 생성 중 오류: {e}")
        return "시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

# 4. 개선된 전체 흐름 함수
def answer(query: str, top_k: int = 5) -> str:
    """사용자 질문에 대한 최종 답변 생성"""
    
    if not query or query.strip() == "":
        return "질문을 입력해 주세요."
    
    try:
        # 검색 단계
        chunks = retrieve_chunks(query, k=top_k)
        
        print(f"📊 검색된 문서 수: {len(chunks)}")
        
        # 답변 생성 단계
        answer_text = generate_answer(query, chunks)
        
        return answer_text
    
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return "죄송합니다. 시스템 오류가 발생했습니다. 관리자에게 문의해 주세요."

# 5. 디버깅용 상세 검색 함수
def debug_search(query: str, top_k: int = 5) -> dict:
    """검색 과정을 상세히 보여주는 디버깅 함수"""
    
    processed_query = preprocess_query(query)
    enhanced_query = expand_query(processed_query)
    
    results = vector_db.similarity_search_with_score(enhanced_query, k=top_k)
    
    debug_info = {
        "original_query": query,
        "processed_query": processed_query,
        "enhanced_query": enhanced_query,
        "results_count": len(results),
        "results": []
    }
    
    for doc, score in results:
        debug_info["results"].append({
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "similarity_score": score,
            "metadata": doc.metadata
        })
    
    return debug_info

# 6. 테스트 함수
if __name__ == "__main__":
    test_queries = [
        "QR 못 찍으면 어떡해?",
        "출석 정정 어떻게 신청해?", 
        "스크린샷 찍어서 제출하면 되나요?",
        "지각했는데 출석 인정받을 수 있어?"
    ]
    
    print("🧪 테스트 시작...")
    for query in test_queries:
        print(f"\n" + "="*50)
        print(f"질문: {query}")
        print("-"*50)
        result = answer(query)
        print(f"답변: {result}")