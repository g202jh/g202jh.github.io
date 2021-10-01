## 기본 설정


```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# 어레이 데이터를 csv 파일로 저장하기
def save_data(fileName, arrayName, header=''):
    np.savetxt(fileName, arrayName, delimiter=',', header=header, comments='')
```

# 과제

## 과제 1
조기 종료를 사용한 배치 경사 하강법으로 로지스틱 회귀를 구현하라.
단, 사이킷런을 전혀 사용하지 않아야 한다.

__단계 1: 데이터 준비__ 

붓꽃 데이터셋의 꽃잎 길이와 꽃잎 너비 특성만 이용한다.

꽃의 품종이 버지니카인지 아닌지 판별할 것이다.


```python
from sklearn import datasets
iris = datasets.load_iris()
```


```python
iris.target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = (iris["target"] == 2).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)
```

0번특성값을 x0이라 판단하기때문에 편향을 추가해야한다.


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]
```

결과를 일정하게 유지하기 위해 랜덤 시드를 지정한다.


```python
np.random.seed(2042)
```

__단계 2: 데이터셋 분할__ 

데이터셋을 훈련, 검증, 테스트 용도로 6대 2대 2의 비율로 무작위로 분할한다.

- 훈련 세트: 60%
- 검증 세트: 20%
- 테스트 세트: 20%

아래 코드는 사이킷런의 train_test_split() 함수를 사용하지 않고 
수동으로 무작위 분할하는 방법을 보여준다.
먼저 각 세트의 크기를 결정한다.


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```

np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다.


```python
rnd_indices = np.random.permutation(total_size)
```

인덱스가 무작위로 섞였기 때문에 무작위로 분할하는 효과를 얻는다. 방법은 섞인 인덱스를 이용하여 지정된 6:2:2의 비율로 훈련, 검증, 테스트 세트로 분할하는 것이다.


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
y_train[:5]
```




    array([0, 0, 1, 0, 0])



__단계 3: 로지느틱 모델 구현__ 

먼저 로지스틱에 사용되는 시그모이드 함수를 만든다.

$$
\sigma(t) = \frac{1}{1 + e^{-t}}
$$


```python
def logistic(logits):
    return 1.0 / (1 + np.exp(-logits))
```

__단계 4: 경사하강법 활용 훈련__ 

가중치를 조정해나가기 위한 세타를 생성한다. 초기값은 랜덤이다.

여기에서 n은 특성이 두개이므로 2가된다.

$$
\begin{align*}
\hat y^{(i)} & = \theta^{T}\, \mathbf{x}^{(i)} \\
 & = \theta_0 + \theta_1\, \mathbf{x}_1^{(i)} + \cdots + \theta_n\, \mathbf{x}_n^{(i)}
\end{align*}
$$


```python
n_inputs = X_train.shape[1] #편향과 특성의 갯수
Theta = np.random.randn(n_inputs) #편향과 특성의 갯수만큼 세타값 랜덤초기화
```

**비용함수 구현**

$$
J(\boldsymbol{\theta}) = -\dfrac{1}{m} \sum\limits_{i=1}^{m}{\left[ y^{(i)} log\left(\hat{p}^{(i)}\right) + (1 - y^{(i)}) log\left(1 - \hat{p}^{(i)}\right)\right]}
$$

위의 수식을 코드로 표현하면 다음과 같다.

-np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))

배치 경사하강법 훈련은 아래 코드를 통해 이루어진다.

- `eta = 0.01`: 학습률
- `n_iterations = 5001` : 에포크 수
- `m = len(X_train)`: 훈련 세트 크기, 즉 훈련 샘플 수
- `epsilon = 1e-7`: $\log$ 값이 항상 계산되도록 더해지는 작은 실수
- `logits`: 모든 샘플에 대한 클래스별 점수, 즉 $\mathbf{X}_{\textit{train}}\, \Theta$
- `Y_proba`: 모든 샘플에 대해 계산된 클래스 별 소속 확률, 즉 $\hat P$


```python
#  배치 경사하강법 구현
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
   
    if iteration % 500 == 0:
      loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
      print(iteration, loss)
    
    error = Y_proba - y_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta = Theta - eta * gradients
```

    0 72.31433784735295
    500 27.030455650044548
    1000 21.92039092375875
    1500 19.414367490717755
    2000 17.784333929402642
    2500 16.59046456120123
    3000 15.657893477152111
    3500 14.899463443628596
    4000 14.265311383111577
    4500 13.724152836628216
    5000 13.25503061135244
    

학습된 파라미터는 다음과 같다.


```python
Theta
```




    array([-10.46790049,   0.48407297,   4.92457437])



**검증**

위에서 얻은 세타값을 가지고
검증세트로 모델 성능을 판단한다.

Y_proba값이 0.5 이상이라면 버지니아로, 아니라면 버지니아가 아니라고 입력해준다.


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.9666666666666667



직접 데이터를 구해보면


```python
y_predict
```




    array([0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.])




```python
y_valid
```




    array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 1])



__단계 5: 규제가 추가된 경사하강법 활용 훈련__ 


```python
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5        # 규제 하이퍼파라미터

Theta = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - y_train
    l2_loss_gradients = np.r_[np.zeros([1]), alpha * Theta[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta = Theta - eta * gradients
```

    0 91.6491324111053
    500 36.52701836668666
    1000 34.36180520510085
    1500 34.03184397960547
    2000 33.97315862139733
    2500 33.96244395839734
    3000 33.96047838771422
    3500 33.96011749640785
    4000 33.96005122388757
    4500 33.96003905353162
    5000 33.9600368185425
    

검증세트에 규제를 적용하니 정확도가 미미하게 떨어졌다.


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.8333333333333334



점수가 조금 떨어졌으나 중요한것은 테스트세트에 대한 성능이다.

__단계 6: 조기 종료 추가__

조기종료는 검증세트에 대한 손실값이 이전 단계보다 커지면
바로 종료되는 기능이다. 이를 코드로 구현하면 다음과 같다.


```python
eta = 0.1 
n_iterations = 50000
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    error = Y_proba - y_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta)
    Y_proba = logistic(logits)
    xentropy_loss = -np.mean(np.sum((y_valid*np.log(Y_proba + epsilon) + (1-y_valid)*np.log(1 - Y_proba + epsilon))))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 55.798959608513385
    500 12.601599687952294
    1000 11.963245890968011
    1500 11.866885229446192
    2000 11.849781121717271
    2500 11.846659454024135
    3000 11.84608683356437
    3500 11.845981698130752
    4000 11.845962391558908
    4500 11.84595884608108
    5000 11.845958194982316
    5500 11.845958075413087
    6000 11.845958053455128
    6500 11.845958049422714
    7000 11.845958048682197
    7500 11.845958048546207
    8000 11.845958048521236
    8416 11.845958048516984
    8417 11.845958048516984 조기 종료!
    

__단계 7: 테스트 세트 평가__


```python
logits = X_test.dot(Theta)
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)


accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9666666666666667



## 과제 2

과제 1에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현하라. 단, 사이킷런을 전혀 사용하지 않아야 한다.

과제2를 수행하기 위해서는 로지스틱 모델을 2개를 사용해야한다.

먼저 setosa인지 아닌지를 판단하는 모델

그리고 virginica인지 아닌지를 판단하는 모델을 각각 만든후에

versicolor일 확률 = '1 - setosa일 확률 - virginica일 확률'로 계산해준다.


```python
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]
y0 = (iris["target"] == 0).astype(np.int) #setosa 판단 모델을 위한 데이터셋
y1 = (iris["target"] == 2).astype(np.int) #virginica 판단 모델을 위한 데이터셋
```


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X] #편향추가
```


```python
np.random.seed(2042) #일정한 결과를 위해 랜덤시드 지정
```


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```


```python
rnd_indices = np.random.permutation(total_size) #데이터 섞기
```

모델 훈련은 각 클래스에 대해 각각 이루어지기 때문에 데이터셋도 각각 준비해준다.


```python
X_train = X_with_bias[rnd_indices[:train_size]] 
y_train = y[rnd_indices[:train_size]]
y_train0 = y0[rnd_indices[:train_size]] #setosa에 대한 라벨
y_train1 = y1[rnd_indices[:train_size]] #virginica에 대한 라벨

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
y_valid0 = y0[rnd_indices[train_size:-test_size]] #setosa에 대한 검증세트 라벨
y_valid1 = y1[rnd_indices[train_size:-test_size]] #virginica에 대한 검증세트 라벨

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
n_inputs = X_train.shape[1]
Theta0 = np.random.randn(n_inputs) #setosa 판단모델에 쓰이는 세타값
Theta1 = np.random.randn(n_inputs) #virginica 판단모델에 쓰이는 세타값
```

**setosa 판별 로지스틱 회귀 모델**


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss0 = np.infty   # 최소 손실값 기억 변수

Theta0 = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits0 = X_train.dot(Theta0)
    Y_proba0 = logistic(logits0)
    error = Y_proba0 - y_train0
    gradients0 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta0[1:]]
    Theta0 = Theta0 - eta * gradients0

    # 검증 세트에 대한 손실 계산
    logits0 = X_valid.dot(Theta0)
    Y_proba0 = logistic(logits0)
    xentropy_loss0 = -np.mean(np.sum((y_valid0*np.log(Y_proba0 + epsilon) + (1-y_valid0)*np.log(1 - Y_proba0 + epsilon))))
    l2_loss0 = 1/2 * np.sum(np.square(Theta0[1:]))
    loss0 = xentropy_loss0 + alpha * l2_loss0
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss0)
        
    # 에포크마다 최소 손실값 업데이트
    if loss0 < best_loss0:
        best_loss0 = loss0
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss0)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss0, "조기 종료!")
        break
```

    0 20.540019459712514
    500 7.7445716153439585
    1000 7.672989036271926
    1500 7.668592640555666
    2000 7.668314272027709
    2500 7.668296612120626
    3000 7.668295491624584
    3500 7.668295420530143
    4000 7.668295416019262
    4500 7.668295415733051
    5000 7.668295415714894
    

**virginica 판별 로지스틱 회귀 모델**


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss1 = np.infty   # 최소 손실값 기억 변수

Theta1 = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits1 = X_train.dot(Theta1)
    Y_proba1 = logistic(logits1)
    error = Y_proba1 - y_train1
    gradients1 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta1[1:]]
    Theta1 = Theta1 - eta * gradients1

    # 검증 세트에 대한 손실 계산
    logits1 = X_valid.dot(Theta1)
    Y_proba1 = logistic(logits1)
    xentropy_loss1 = -np.mean(np.sum((y_valid1*np.log(Y_proba1 + epsilon) + (1-y_valid1)*np.log(1 - Y_proba1 + epsilon))))
    l2_loss1 = 1/2 * np.sum(np.square(Theta1[1:]))
    loss1 = xentropy_loss1 + alpha * l2_loss1
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss1)
        
    # 에포크마다 최소 손실값 업데이트
    if loss1 < best_loss1:
        best_loss1 = loss1
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss1)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss1, "조기 종료!")
        break
```

    0 45.3881848638996
    500 12.482904005693054
    1000 11.947222069327108
    1500 11.864096195806567
    2000 11.849273910674974
    2500 11.846566475123907
    3000 11.846069764314986
    3500 11.845978563684064
    4000 11.845961815948371
    4500 11.845958740374874
    5000 11.845958175570196
    

**이제 테스트셋에 적용해본다.**

위에서 구한 두개의 세타값을 이용해

1.   setosa일 확률(setosa_proba)
2.   virginica일 확률(virginica_proba)
3.   versicolor일 확률(1 - setosa_proba - virginica_proba)

셋중에 가장 높은것을 채택하여 분류를 진행한다.


```python
logits = X_test.dot(Theta0) #setosa에 대한 확률값 추정  
setosa_proba = logistic(logits)

logits = X_test.dot(Theta1) #virginica에 대한 확률값 추정 
virginica_proba = logistic(logits)

y_predict = np.array([])
for i in range(len(Y_proba0)):
  prob_list = [[setosa_proba[i], 0], [1-setosa_proba[i]-virginica_proba[i], 1], [virginica_proba[i], 2]]
  prob_list.sort(reverse=True) #가장 높은 확률이 가장 앞으로 오게끔 정렬해준다.
  y_predict = np.append(y_predict, prob_list[0][1]) #가장 확률이 높았던 것을 예측값으로 결정한다.
```


```python
accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9333333333333333


