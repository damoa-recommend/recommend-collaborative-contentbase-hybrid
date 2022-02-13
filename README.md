### 컨텐츠 기반 필터링(content based filtering)

사용자의 이전 행동과 명시적 피드백을 통해 사용자가 좋아하는 것과 유사한 항목을 추천

* 장점

```
많은 수의 사용자를 대상으로 쉽게 확장 가능
사용자가 없더라도 추천 가능
개인 맞춤형 추천 가능
```

* 단점

```
도메인 지식이 많이 필요
사용자의 기존 관심사항을 기반으로만 추천 가능 -> 사용자의 이력이 없다면 추천 불가능
```

### 협업 필터링(collaborative filtering)

사용자와 항목간의 유사성을 동시에 사용해서 추천

기존에 내 관심사가 아닌 항목이라도 추천 가능
자동으로 임베딩 학습 가능

협업 필터링 종률로 **`사용자 기반 협업 필터링`**, **`아이템 기반 협업 필터링`** 존재.

* 장점

```
자동으로 임베딩을 학습하기 때문에 도메인 지식이 필요 없다.
기존의 관심사가 아니더라도 추천 가능
```

* 단점

```
학습 과정에 나오지 않은 항목은 임베딩을 만들 수 없음
추가 특성을 사용하기 어려움
```

# configuration

### dependencies

```
scikit-surprise
sklearn
nltk
```

```
$ poetry install
```

```
$ python content-based-filtering.py

$ python collaborative-filtering.py

$ python hybrid.py
```