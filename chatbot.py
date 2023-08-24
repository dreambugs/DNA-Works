#Setup key
import os

os.environ["OPENAI_API_KEY"] = "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" # 환경변수에 OPENAI_API_KEY를 설정합니다.
from langchain.document_loaders import PyPDFLoader # PDF 문서를 로드하기 위한 도구
from langchain.embeddings.openai import OpenAIEmbeddings # OpenAIEmbeddings는 OpenAI를 기반으로 한 텍스트 임베딩을 생성하는 클래스
from langchain.embeddings.cohere import CohereEmbeddings # CohereEmbeddings는 Cohere 기반의 텍스트 임베딩을 생성하는 클래스입니다.
from langchain.text_splitter import CharacterTextSplitter # 텍스트를 문자 단위로 분리하는 도구
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch # ElasticSearch 기반의 벡터 검색 도구
from langchain.vectorstores import Chroma # Chroma는 벡터 저장과 관련된 여러 기능을 제공하는 모듈

# PDF를 벡터DB에 등록해서, 이를 기반으로 채팅해보기
loader = PyPDFLoader("호남연수원.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings() # openai의 임베딩 모델 가져오기
vector_store = Chroma.from_documents(texts, embeddings) # 수치화해서 집어넣자(주어진 텍스트에 대한 임베딩을 생성해서 벡터저장소에 저장) -> embeddings
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Chroma 벡터 저장소를 기반으로 검색가능한 검색기(retriever)를 생성합니다. 'k': 2는 검색 결과로 상위 2개의 항목을 반환하도록 설정합니다.

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain # 소스 정보와 함께 검색 기반의 질의응답(QA) 작업을 수행하는 체인(파이프라인)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

chain = RetrievalQAWithSourcesChain.from_chain_type( #문서 source와 함께 검색 기능 질의응답 Chain
    llm=llm,
    retriever = retriever, # 검색기
    return_source_documents=True) # 결과 얻을 때 source까지

query = "호남연수원 설립취지 알려줘"
result = chain(query)
result['answer']

#////////////////////////////////////////////////////////////

from langchain.prompts.chat import (
    ChatPromptTemplate, # ChatPromptTemplate는 채팅에 사용되는 기본 템플릿
    SystemMessagePromptTemplate, # SystemMessagePromptTemplate는 시스템 메시지를 위한 템플릿
    HumanMessagePromptTemplate, # HumanMessagePromptTemplate는 사용자(인간) 메시지를 위한 템플릿
)

system_template="""Use the following pieces of context to answer the users question shortly.
Your task is to answer based on the documents from the 호남연수원.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}

You MUST answer in Korean:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

chain_type_kwargs = {"prompt": prompt}

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)  # Modify model_name if you have access to GPT-4

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
import gradio as gr

def respond(message, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.
    # global previous_message
    result = chain(message)

    bot_message = result['answer']

    # for i, doc in enumerate(result['source_documents']):
    #     bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + '(' + str(doc.metadata['page']) + ') '
    # previous_message += messages
    chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.

    return "", chat_history  # 수정된 채팅 기록을 반환합니다.

with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
    chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
    msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
    clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

    msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

demo.launch(debug=True)  # 인터페이스를 실행합니다. 실행하면 사용자는 '입력' 텍스트박스에 메시지를 작성하고 제출할 수 있으며, '초기화' 버튼을 통해 채팅 기록을 초기화 할 수 있습니다.

print(texts)

