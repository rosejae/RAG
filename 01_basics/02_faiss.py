import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

#
# chunking
#

text = """생성형 인공지능 또는 생성형 AI는 프롬프트에 대응하여 텍스트, 이미지, 기타 미디어를 생성할 수 있는 일종의 인공지능 시스템이다.
단순히 기존 데이터를 분석하는 것이 아닌, 새로운 콘텐츠를 만드는 데 초점을 맞춘 인공지능 분야를 말한다. 2022년경부터 본격적으로 유명해지기 시작했다.
데이터 원본을 통한 학습으로 소설, 이미지, 비디오, 코딩, 음악, 미술 등 다양한 콘텐츠 생성에 이용된다. 한국에서는 2022년 Novel AI 등, 그림 인공지능의 등장으로 주목도가 높아졌으며, 해외에서는 미드저니, 챗GPT등 여러 모델을 잇달아 공개하면서 화제의 중심이 되었다.
보통 딥러닝 인공지능은 학습 혹은 결과 출력 전 원본 자료를 배열 자료형[2] 숫자 데이터로 변환하는 인코딩 과정이 중요한데, 생성 AI의 경우 인공지능의 출력 데이터를 역으로 그림, 글 등의 원하는 형태로 변환시켜주는 디코딩 과정 또한 필요하다.
사실상 인공지능의 대중화를 이끈 기술로써, 해당 기술이 인공지능에 대한 사람들의 전반적인 인식을 매우 크게 바꿔놨다고 해도 과언이 아니다.
"""

# CharacterTextSplitter settings
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    )

# chunking
chunks = splitter.split_text(text)

for sentence in chunks:
    print(f"chunked sentence: {sentence}")

#
# embedding model, vector database
#

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002", 
    api_key=os.getenv("OPENAI_API_KEY"),
    )

# 텍스트 임베딩을 FAISS FAISS(Vector Store)에 저장
knowledge_base = FAISS.from_texts(chunks, embeddings)

# similarity
question = "생성형 AI란?"
references = knowledge_base.similarity_search(question)

#
# LLM
#

llm = ChatOpenAI(model="gpt-4o", 
                 api_key=os.getenv("OPENAI_API_KEY"),
                 )

# Q&A Interface, 하지만 모든 document를 가져온다
chain = load_qa_chain(llm, chain_type="stuff")

with get_openai_callback() as cb:
    response = chain.run(input_documents=references, question=question)
    print(cb)

for line in response.split(". "):
    print(line)