import requests
from dotenv import load_dotenv
import os
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.exceptions import CouchbaseException
import couchbase.search as search
from couchbase.vector_search import VectorQuery, VectorSearch
import openai
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONN_STR = os.getenv("DB_CONN_STR")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_BUCKET = os.getenv("DB_BUCKET")
DB_SCOPE = os.getenv("DB_SCOPE")
DB_COLLECTION = os.getenv("DB_COLLECTION")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Couchbase 클러스터 연결 설정
auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
cluster = Cluster.connect(DB_CONN_STR, ClusterOptions(auth))
bucket = cluster.bucket(DB_BUCKET)
scope = bucket.scope(DB_SCOPE)
collection = scope.collection(DB_COLLECTION)
article_index_name = INDEX_NAME  # 인덱스 이름 설정

# 텍스트를 벡터로 변환하는 함수 (Clova API 사용)
def generate_vector_with_openai(query_text):
    """
    주어진 텍스트를 Clova API를 통해 임베딩 벡터로 변환합니다.
    :param query_text: 벡터화할 텍스트
    :return: 벡터 리스트
    """
    
    try:
        # 텍스트 임베딩 생성
        response = client.embeddings.create(input = [query_text], model=EMBEDDING_MODEL).data[0].embedding
        return response
   
    except openai.RateLimitError as e:
        # API 호출 한도 초과 처리 (rate limit exceeded)
        print(f"Rate limit exceeded: {e}..")
        return None
        
    except openai.APIError as e:
        # 서버 측 오류 처리
        print(f"OpenAI API error: {e}. ")
        return None
    
    except openai.AuthenticationError as e:
        # 인증 오류 처리 (API 키가 잘못되었을 때)
        print(f"Authentication error: {e}. Check your API key.")
        return None
     
    except openai.APIConnectionError as e:
        # 네트워크 오류 처리 
        print(f"Network error: {e}. ")
        # time.sleep(5)  # 5초 대기 후 재시도
        return None

    except Exception as e:
        # 기타 예외 처리
        print(f"Unexpected error: {e}. ")
        return None

# 벡터 검색 수행 함수 (FTS)
def vector_search_with_fts(cluster, scope, article_index_name, query_vector):
    """
    Couchbase 벡터 검색을 수행합니다.
    :param cluster: Couchbase 클러스터
    :param scope: Couchbase 스코프
    :param article_index_name: FTS 인덱스 이름
    :param query_vector: 검색할 벡터
    """
    try:
        # 벡터 검색 쿼리 설정
        vector_search = VectorSearch.from_vector_query(VectorQuery('article_vector', query_vector, num_candidates=5))
        
        request = search.SearchRequest.create(vector_search)

        # 검색 수행
        result = scope.search(article_index_name, request)

        print(f"FTS Vector Search results:")
        for row in result.rows():
            print(f"ID: {row.id}, Score: {row.score}")
            doc = collection.get(row.id)
            doc_content = doc.content_as[dict]  # 문서를 사전 형식으로 변환
            print(f"Title: {doc_content['title']}")
            print(f"Date: {doc_content['date']}")
            print(f"Url: {doc_content['url']}")
            print("--------")

    except CouchbaseException as e:
        print(f"Search failed: {e}")

# SQL++ 하이브리드 검색 수행 함수
def hybrid_vector_search_with_sql(cluster, article_vector, title_vector, title_text):
    """
    Couchbase SQL++, 자연어검색, 벡터 검색을 결합하여 검색을 수행합니다.
    :param cluster: Couchbase 클러스터
    :param article_vector: 검색할기사 내용 벡터
    :param title_vector: 검색할기사 제목 벡터
    :param title_text: 검색할 단어
    """
    try:
        # N1QL을 사용한 KNN 및 필터 검색
        query = f"""
        SELECT title, date, author, url, like_count, SEARCH_SCORE() AS score
        FROM `vectordemo`.semantic.article AS t1
        WHERE author like "%기자"
        AND like_count >= 1
	AND SEARCH(t1, {{
                "query": {{"match": "{title_text}","field":"title"}}
            }})
        AND SEARCH(t1, {{
                "query": {{"match_none": {{}}}},
                "knn": [{{"field": "article_vector", "vector": {article_vector}, "k": 5}}],
                "knn": [{{"field": "title_vector", "vector": {title_vector}, "k": 5}}]
            }})
        ORDER BY score,date DESC
        """
        
        # 쿼리 실행
        result = cluster.query(query)
        
        # 결과 출력
        print("")
        print(f"SQL++ Hybrid Search results:", result)
        for row in result:
            print(f"Score: {row['score']}")
            print(f"Title: {row['title']}")
            print(f"Date: {row['date']}")
            print(f"Author: {row['author']}")
            print(f"Like Count: {row['like_count']}")
            print(f"Url: {row['url']}")
            print("--------")
    except CouchbaseException as e:
        print(f"Hybrid search failed: {e}")

# 메인 함수
def main():

    # 검색할 텍스트 입력
    article_text = input("Enter text to vector search in article : ")
    title_text = input("Enter text to test search in title : ")

    # 텍스트를 벡터로 변환
    article_vector = generate_vector_with_openai(article_text)
    title_vector = generate_vector_with_openai(title_text)

    if not article_vector:
        print("No vector generated, exiting search.")
        return

    # FTS 벡터 검색 수행
    vector_search_with_fts(cluster, scope, article_index_name, article_vector)

    # SQL++ 하이브리드 검색 수행
    hybrid_vector_search_with_sql(cluster, article_vector, title_vector, title_text)

# 메인 함수 실행
if __name__ == "__main__":
    main()
