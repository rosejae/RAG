import faiss 
import numpy as np 

np.random.seed(1)  # 일관된 결과를 얻기 위해 시드 설정 

dimension = 128    # 각 벡터의 차원                
n = 200            # 벡터 수  
nlist = 5          # 클러스터(셀이라고도 표현) 수                 

#
# data
#
            
db_vectors = np.random.random((n, dimension)).astype('float32') #(200 * 128) 벡터 행렬

#
# FAISS index train
#

quantiser = faiss.IndexFlatL2(dimension)                                  #quantizer를 활용해서 클러스터 생성
index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)  #Inverted File 만들기
#index = faiss.IndexPQ(D, m, nbits)
#index = faiss.IndexIVFPQ(vecs, D, nlist, m, nbits)

print(f'index train mode: {index.is_trained}')   # False
index.train(db_vectors)                          # 데이터베이스 벡터에 대한 훈련
print(f'the number of nodes: {index.ntotal}')

index.add(db_vectors)                            # 벡터를 추가하고 인덱스를 업데이트
print(f'index train mode: {index.is_trained}')   # True
print(f'the number of nodes: {index.ntotal}')

#
# search
#

# nprobe = 2       # 가장 유사한 클러스터 2개 찾기
n_query = 10       # 10개의 쿼리 벡터
k = 3              # 가장 가까운 이웃 3개를 반환

query_vectors = np.random.random((n_query, dimension)).astype('float32') # 쿼리 벡터 생성
distances, indices = index.search(query_vectors, k)                      # 검색 수행

#
# save
#

faiss.write_index(index, "vector.index")  # 인덱스를 디스크에 저장
index = faiss.read_index("vector.index")  # 인덱스를 로드