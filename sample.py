from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k', prompt=False) # [('196', '242', 3.0, '881250949')] 사용자, 영화, 평점, 아이디
print('[data.raw_ratings]')
print(data.raw_ratings[:10]) 

model = SVD()

# TODO: 머하는 녀석인지 공부하기!
# cross_validate(model, data, measures=['rmse', 'mae'], cv =5, verbose=True)
