from langchain.tools import tool
from langchain_elasticsearch import (
    ElasticsearchStore,
    BM25Strategy,
    DenseVectorStrategy,
)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="qwen3-embedding",
    base_url="http://localhost:8082/v1",
    api_key="your_api_key",
)
vector_store = ElasticsearchStore(
    embedding=embeddings,
    index_name="236",  # 索引名
    es_url="http://localhost:9200",  # 或 es_cloud_id, es_user, es_password 等
    es_user="elastic",
    es_password="elastic@2024",
    strategy=DenseVectorStrategy(),
)
bm25_vector_store = ElasticsearchStore(
    embedding=embeddings,
    index_name="236",  # 索引名
    es_url="http://localhost:9200",  # 或 es_cloud_id, es_user, es_password 等
    es_user="elastic",
    es_password="elastic@2024",
    strategy=BM25Strategy(),
)
query = "等节水灌溉方式。\n水资源短缺地区应当严格控制人造河湖等景观用水"


@tool
def retrieve_context(query: str):
    """检索与查询相关的信息。"""
    docs = vector_store.similarity_search_with_score(query, k=3)  # 检索 top-3 文档
    # vector_store.similarity_search_with_relevance_scores(query=query)
    context = ""
    for idx, (doc, socre) in enumerate(docs, start=1):
        context += f"文档 {idx}: \n{doc.page_content}\n\n"
    return context


retrieve_tool = retrieve_context

if __name__ == "__main__":

    r = retrieve_context.invoke(query)
    print(r)
