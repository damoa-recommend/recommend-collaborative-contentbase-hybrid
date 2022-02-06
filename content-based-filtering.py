'''
유사도 기반으로 추천

장점: 많은 수의 사용자를 대상으로 쉽게 확장 가능
     사용자가 관심을 갖지 않던 상품 추천 가능

단점: 도메인 지식이 많이 필요
     사용자의 기존 관심사항을 기반으로만 추천 가능
'''

import numpy as np
from surprise import Dataset

def get_recommend(my_vector, best_match_vector):
  # my_id와 유사한 유저가 본 영화중 my_id가 보지않은 영화 추출
  recommend_movies = []
  for idx, log in enumerate(zip(my_vector, best_match_vector)):
    my_rating_log, best_rating_match_log = log
    if my_rating_log == 0 and best_rating_match_log != 0:
      recommend_movies.append(idx)
  return recommend_movies

data = Dataset.load_builtin('ml-100k', prompt=False) # [('196', '242', 3.0, '881250949')] 사용자, 영화, 평점, 시간
raw_data = np.array(data.raw_ratings, dtype=int)

# 첫 번째 두 번째 컬럼은 사용자와 영화의 유니크 값인 ID를 의미한다. 기존 데이터는 1부터 시작하므로 1을 빼서 0부터 시작하도록 변경
raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])
shape = (n_users + 1, n_movies + 1) # (943, 1682)

adj_matrix = np.ndarray(shape, dtype=int) # shape에 맞춰서 0으로 채워줌

for user_id, movie_id, rating, time in raw_data:
  # TODO: 만약, 평가 점수가 아닌 1을 넣는다면 봤는지 안봤는지에 대해서 추천을 시도
  # rating값을 넣으면 평가 점수를 기반으로 추천을 수행함
  adj_matrix[user_id][movie_id] = rating 

print(adj_matrix)

# 방법 1. my_id를 가진 사용자와 가장 유사한 유저 찾기
my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(adj_matrix):
  if my_id == user_id:
    continue
  else :
    similarity = np.dot(my_vector, user_vector) # 내적이 크면 유사도가 높다.
    if similarity > best_match:
      best_match = similarity
      best_match_id = user_id
      best_match_vector = user_vector

print()
print('내적을 이용한 컨텐츠 기반 추천')
print(best_match, best_match_id, best_match_vector)

way1_recommend = get_recommend(my_vector, best_match_vector)
print(way1_recommend[: 5])

# 방법 2. my_id를 가진 사용자와 가장 유사한 유저 찾기(유클리드 거리)
my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = 9999, -1, []

for user_id, user_vector in enumerate(adj_matrix):
  if my_id == user_id:
    continue
  else :
    euclidean_dist = np.sqrt(np.sum(np.square(my_vector - user_vector))) # 유클리드 거리는 가까울 수록 유사도가 높다
    if euclidean_dist < best_match:
      best_match = euclidean_dist
      best_match_id = user_id
      best_match_vector = user_vector

print()
print('유클리드 거리를 이용한 컨텐츠 기반 추천')
print(best_match, best_match_id, best_match_vector)

way2_recommend = get_recommend(my_vector, best_match_vector)
print(way2_recommend[: 5])

# 방법 3. 코사인 유사도를 이용한 추천
def compute_cos_similarity(v1, v2):
  norm1 = np.sqrt(np.sum(np.square(v1)))
  norm2 = np.sqrt(np.sum(np.square(v2)))
  return np.dot(v1, v2) / (norm1 * norm2)

my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(adj_matrix):
  if my_id == user_id:
    continue
  else :
    cos_similarity = compute_cos_similarity(my_vector, user_vector) # 코사인 유사도는 값이 클수록 유사도가 높다
    if cos_similarity > best_match:
      best_match = cos_similarity
      best_match_id = user_id
      best_match_vector = user_vector

print()
print('코사인 유사도 이용한 컨텐츠 기반 추천')
print(best_match, best_match_id, best_match_vector)

way2_recommend = get_recommend(my_vector, best_match_vector)
print(way2_recommend[: 5])