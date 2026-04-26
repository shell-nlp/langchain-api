from langchain.tools import tool

from langchain_api.rag.elastic_graph_rag import ElasticGraphRAG
from langchain_api.rag.elastic_utils import Elasticsearch
from langchain_api.settings import settings
from langchain_api.utils import get_embedding_model

DEFAULT_INDEX_NAME = "236"

embeddings = get_embedding_model()
es_retriever = Elasticsearch(
    url=settings.ES_URL,
    username=settings.ES_URSR,
    password=settings.ES_PWD,
    embedding_model=embeddings,
)


@tool
def retrieve_context(query: str):
    """检索与查询相关的信息。"""
    docs = es_retriever.retrieve(query=query, k=3, index_name=DEFAULT_INDEX_NAME)
    context = ""
    for idx, doc in enumerate(docs, start=1):
        context += f"文档 {idx}: \n{doc.get('content', '')}\n\n"
    return context


retrieve_tool = retrieve_context


@tool
def retrieve_graph_context(query: str, graph_name: str = DEFAULT_INDEX_NAME):
    """使用 ES 向量图 RAG 检索与查询相关的上下文。"""
    es = Elasticsearch(
        url=settings.ES_URL,
        username=settings.ES_URSR,
        password=settings.ES_PWD,
        embedding_model=embeddings,
    )
    rag = ElasticGraphRAG(es=es, graph_name=graph_name)
    result = rag.retrieve(query=query, k=5)
    passages = result["passages"] if isinstance(result, dict) else result

    context = ""
    for idx, doc in enumerate(passages, start=1):
        context += f"文档 {idx}: \n{doc.get('content', '')}\n\n"
    return context


if __name__ == "__main__":
    query = "等节水灌溉方式。\n水资源短缺地区应当严格控制人造河湖等景观用水"

    r = retrieve_context.invoke(query)
    print(r)
