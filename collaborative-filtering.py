'''
사용자와 항목의 유사성을 동시에 고려해 추천
기존에 내 관심사가 아닌 항목이라도 추천 가능
자동으로 임베딩 학습 가능

장점: 자동으로 임베딩을 학습하기 때문에 도메인 지식이 필요 없다.
    기존의 관심사가 아니더라도 추천 가능

단점: 학습 과정에 나오지 않은 항목은 임베딩을 만들 수 없음
     추가 특성을 사용하기 어려움
'''

from surprise import KNNBasic, SVD, SVDpp, NMF
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k', prompt=False) # [('196', '242', 3.0, '881250949')] 사용자, 영화, 평점, 시간

model = KNNBasic()
cross_validate(model, data, measures=['rmse', 'mae'], cv =5, verbose=True, n_jobs=4)

model = SVD()
cross_validate(model, data, measures=['rmse', 'mae'], cv =5, verbose=True, n_jobs=4)

model = SVDpp()
cross_validate(model, data, measures=['rmse', 'mae'], cv =5, verbose=True, n_jobs=4)

model = NMF()
cross_validate(model, data, measures=['rmse', 'mae'], cv =5, verbose=True, n_jobs=4)