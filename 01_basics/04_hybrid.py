import os
from dotenv import load_dotenv

from langchain.retrievers import EnsembleRetriever # 여러 retriever를 입력으로 받아 처리
from langchain_community.retrievers import BM25Retriever  #TF-IDF 계열의 검색 알고리즘
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

"""
metadata는 단지 얘가 뭘 사용했는지 구분자를 나타냄
source 1은 bm25이고, source 2는 faiss다
"""

#
# bm25 retriever
#

doc_list_1 = [
    "프렌치 불독: 사교적이고 친근한 성격을 가지고 있으며, 조용하고 집에서 지내기에 적합 합니다",
    "비글: 호기심이 많고, 에너지가 넘치며, 사냥 본능이 강합니다. ",
    "독일 셰퍼드: 용감하고 지능적이며, 충성심이 강합니다",
    "포메라니안: 활발하고 호기심이 많으며, 주인에게 매우 애정적입니다",
    "치와와: 작지만 용감하고, 주인에게 깊은 애정을 보입니다",
    "보더 콜리:	매우 지능적이고 학습 능력이 뛰어나며, 에너지가 많아 많은 운동이 필요합니다 "
]

bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, 
    metadatas=[{"source": 1}]*len(doc_list_1),
    )
bm25_retriever.k = 2

#
# FAISS
#

doc_list_2 = [
    "프렌치 불독: 열에 약하므로 주의가 필요합니다",
    "비글: 가족과 잘 지내며, 아이들과 노는 것을 좋아합니다.",
    "독일 셰퍼드: 경찰견이나 구조견으로 많이 활용되며, 적절한 훈련과 운동이 필요합니다.",
    "포메라니안: 털이 풍성하므로 정기적인 그루밍이 필요합니다.",
    "치와와: 다른 동물이나 낯선 사람에게는 조심스러울 수 있습니다.",
    "보더 콜리: 목축견으로서의 본능이 강하며, 다양한 트릭과 명령을 쉽게 배울 수 있습니다."
]

# 임베딩 이전에 청킹이 필요하지만, 텍스트가 몇 줄 안되니 그냥 집어넣음
embedding = OpenAIEmbeddings(model="text-embedding-ada-002", 
                             openai_api_key=os.getenv("OPENAI_API_KEY"),
                             )

faiss_vectorstore = FAISS.from_texts(
    doc_list_2, 
    embedding, 
    metadatas=[{"source": 2}]*len(doc_list_2),
    )

faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

#
# ensemble
#

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], 
    weights=[0.5, 0.5], #retriever 가중치 설정
    )

#
# inference
#

query = "충성심이 강한 강아지는?"
ensemble_result = ensemble_retriever.get_relevant_documents(query)
bm25_result = bm25_retriever.get_relevant_documents(query)
faiss_result = faiss_retriever.get_relevant_documents(query)

# 가져온 문서를 출력합니다.
print("[Ensemble Retriever]\n", ensemble_result, end="\n\n")
print("[BM25 Retriever]\n", bm25_result, end="\n\n")
print("[FAISS Retriever]\n", faiss_result, end="\n\n")