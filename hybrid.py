from sklearn.decomposition import randomized_svd, TruncatedSVD
from surprise import Dataset
import numpy as np

def get_recommend(adj_matrix, my_id, best_match_id):
  # my_id와 유사한 유저가 본 영화중 my_id가 보지않은 영화 추출
  recommend_movies = []
  for idx, log in enumerate(zip(adj_matrix[my_id], adj_matrix[best_match_id])):
    my_rating_log, best_rating_match_log = log
    if my_rating_log == 0 and best_rating_match_log != 0:
      recommend_movies.append(idx)
  return recommend_movies

data = Dataset.load_builtin('ml-100k', prompt=False)
raw_data = np.array(data.raw_ratings, dtype=int)

raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])

shape = (n_users + 1, n_movies+ 1)
print(shape)

adj_matrix = np.ndarray(shape, dtype=int) # shape에 맞춰서 0으로 채워줌
for user_id, movie_id, rating, time in raw_data:
  adj_matrix[user_id][movie_id] = rating

print(adj_matrix)

svd = TruncatedSVD(n_components=5, n_iter=5, random_state=None)
svd.fit(adj_matrix)
print(svd.components_) # 특잇값 분해된 행렬을 다시 합친 결과 np.dot(np.dot(U, S), VT)
# LSA 알고리즘에선 이용할 땐 다시 복원한 행렬을 기반으로 행 백터를 정렬하여 상위 n개의 단어가 해당 문서의 주요 단어가 된다.
# LSA 알고리즘 샘플 ===> https://github.com/damoa-recommend/SVD-LSA
# 사용자 기반, 아이템 기반 추천시엔 사용자, 아이템 기반으로 분해된 행렬을 이용하여 내적, 유클리드 거리, 코사인 유사도를 활용하여 추천을 진행한다.

U, S, VT = randomized_svd(adj_matrix, n_components=2, n_iter=5, random_state=None) # 분해
S = np.diag(S)

recovery = np.dot(np.dot(U, S), VT) # 복원 : 앞의 svd.components_와 결과가 동일함
print(recovery.shape)

def compute_cos_similarity(v1, v2):
  norm1 = np.sqrt(np.sum(np.square(v1)))
  norm2 = np.sqrt(np.sum(np.square(v2)))
  return np.dot(v1, v2) / (norm1 * norm2)

# 방법 1. SVD + 코사인 유사도를 이용한 사용자 기반 추천
'''
사용자 기반 추천
나와 비슷한 취향을 가진 다른 사용자의 행동을 추천
사용자 특징 벡터의 유사도 사용(U)
'''
my_id, my_vector = 0, U[0] # SVD의 압축된 유저 정보를 가지고 있는 U 행렬을 이용한다. 이때 열은 기존의 영화의 특성으로 분류된 재정의된 내용을 가지고 있다.
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(U):
  if my_id == user_id:
    continue
  else :
    cos_similarity = compute_cos_similarity(my_vector, user_vector) # 코사인 유사도는 값이 클수록 유사도가 높다
    if cos_similarity > best_match:
      best_match = cos_similarity
      best_match_id = user_id
      best_match_vector = user_vector

print()
print('SVD + 코사인 유사도 이용한 사용자 기반 추천')
print(best_match, best_match_id, best_match_vector)

way1_recommend = get_recommend(adj_matrix, my_id, best_match_id)
print(way1_recommend[: 5])

# 방법 2. SVD + 코사인 유사도를 이용하여 항목기반 추천
'''
항목 기반 추천
내가 본 항목과 비슷한 항목추천
항목 특징 벡터의 유사도 사용(VT)
'''
my_id, my_vector = 0, VT.T[0] # SVD의 압축된 항목(영화) 정보를 가지고 있는 VT 행렬을 이용한다. 해당 행렬은 SVD에 의해 전치되어 있으므로 전치행렬을 이용하여 사용한다.
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(VT.T):
  if my_id == user_id:
    continue
  else :
    cos_similarity = compute_cos_similarity(my_vector, user_vector) # 코사인 유사도는 값이 클수록 유사도가 높다
    if cos_similarity > best_match:
      best_match = cos_similarity
      best_match_id = user_id
      best_match_vector = user_vector

print()
print('SVD + 코사인 유사도 이용한 컨텐츠 기반 추천')
print(best_match, best_match_id, best_match_vector)

way2_recommend = []
for i, user_vector in enumerate(adj_matrix):
  if adj_matrix[i][my_id] > 0.9:
    way2_recommend.append(i)

print(way2_recommend[: 5])
print()