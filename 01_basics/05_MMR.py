from dotenv import load_dotenv
load_dotenv()

texts = [
    """ChatGPT 열풍으로 인해 생성형 AI에 대한 관심이 뜨겁다. 생성형 AI는 이용자의 특정 요구에 따라 결과를 능동적으로 생성해 내는 인공지능 기술이다.""",
    """특히, 생성형 AI는 대량의 데이터(Hyper-scale Data)를 학습하여 인간의 영역이라고 할 수 있는 창작의 영역까지 넘보고 있다.""",
    """베타 버전 출시2개월 만에 MAU(월간 활성 이용자 수)가 무려 1억 명을 넘어섰다. 또한 구글, 메타 등 글로벌 빅테크 기업들이 앞다투어 천문학적인 규모의 투자와 유사 서비스 출시 계획을발표하고 있다.""",
    """이 서비스의 핵심은 서비스 이용자의 질문을 이해하고 분석하여 수많은 정보 중 답이 될 만한 필요정보를 스스로 찾아서 이를 적절히 요약과 정리해 제공하는 것이다 """,
    """특히 앞서 질문한 내용의 맥락을 잇거나 구체적인 사례를 들어 질문할수록 더 정확한 답을 얻을 수 있는데, 이는 마치 사람과 대화하는 것처럼 맥락을 이해하여 답을 제공한다는 점에서 이전과 차원이 다른 정보 검색 서비스를 체감하게 한다."""
]

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
smalldb = Chroma.from_texts(texts, embedding=embedding)

#
# Similarity search
#

question = "생성형 AI 핵심 기능은?"
docs_ss = smalldb.similarity_search(question, k=2)

print(docs_ss[0].page_content[:100])
print('\n\n', docs_ss[1].page_content[:100])

#
# MMR
#

docs_mmr = smalldb.max_marginal_relevance_search(question, k=2)

print(docs_mmr[0].page_content[:100])
print('\n\n',docs_mmr[1].page_content[:100])