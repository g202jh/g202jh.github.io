---
layout: post
title:  "텐서(Tensor)란?"
---

# **텐서 소개**


```python
import tensorflow as tf
import numpy as np
```

**텐서**는 일관된 유형 (dtype이라고 불림)을 가진 다차원 배열이다. 지원되는 모든 `dtypes`은 `tf.dtypes.DType`에서 볼 수 있다.

모든 텐서는 Python 숫자 및 문자열과 같이 변경할 수 없다. 

텐서의 내용을 업데이트 할 수 없으며 새로운 텐서를 만들수만 있다.

## 기초

기본 텐서를 만들어 보자.

다음은 "스칼라" 또는 "rank-0" 텐서이다. 스칼라는 단일 값을 포함하며 "축" 은 없다.


```python
# 이것은 기본적으로 int32 텐서가 됩니다. 아래 "dtypes"를 참조하세요.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```

    tf.Tensor(4, shape=(), dtype=int32)
    

"벡터" 또는 "rank-1" 텐서는 값의 목록과 같다. 벡터에는 하나의 축이 있다.


```python
# 이것을 float 텐서로 만들어 보겠습니다.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
```

    tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
    

"행렬" 또는 "rank-2" 텐서에는 두 개의 축이 있다.


```python
# 특정하게 하려면 생성 시 dtype(아래 참조)을 설정할 수 있습니다.
rank_2_tensor = tf.constant([[1, 2],
                            [3, 4],
                            [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

    tf.Tensor(
    [[1. 2.]
     [3. 4.]
     [5. 6.]], shape=(3, 2), dtype=float16)
    

<tr>
  <th>스칼라, 모양: <code>[]</code> </th>
  <th>벡터, 모양: <code>[3]</code> </th>
  <th>행렬, 모양: <code>[3, 2]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/scalar.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/vector.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/matrix.png)

텐서에는 더 많은 축이 있을 수 있다. 여기서는 3개의 축이 있는 텐서가 사용될 것이다.


```python
# 임의의 수의 축이 있을 수 있습니다. (때로는 "차원"이라고도 함)
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
        
print(rank_3_tensor)
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
    

축이 2개 이상인 텐서를 시각화하는 방법에는 여러 가지가 있다.

<tr>
  <th>3축 텐서, 모양: <code>[3, 2, 5]</code> 
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/3-axis_numpy.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/3-axis_front.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/3-axis_block.png)


np.array 또는 tensor.numpy 메서드를 사용하여 텐서를 NumPy 배열로 변환할 수 있다.


```python
np.array(rank_2_tensor)
```




    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)




```python
rank_2_tensor.numpy()
```




    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)



텐서에는 종종 float과 int가 포함되지만, 다음과 같은 다른 유형도 있다.

*   복소수
*   문자열

기본 `tf.Tensor` 클래스에서는 텐서가 '직사각형'이어야 한다.

즉, 각 축을 따라 모든 요소의 크기가 같다. 

그러나 다양한 형상을 처리할 수 있는 특수 유형의 텐서가 있다.

Ex)
*   비정형 (Ragged) 텐서
*   희소 텐서





덧셈, 요소별 곱셈 및 행렬 곱셈을 포함하여 텐서에 대한 기본 산술을 수행할 수 있다.


```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # `tf.ones([2,2])'라고 말할 수도 있다.

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 
    
    


```python
print(a + b, "\n") # 요소별 덧셈
print(a * b, "\n") # 요소별 곱셈
print(a @ b, "\n") # 행렬 곱셈
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 
    
    

텐서는 모든 종류의 연산(ops)에 사용된다.


```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# 가장 큰 값 찾기
print(tf.reduce_max(c))
# 가장 큰 값의 인덱스 찾기
print(tf.argmax(c))
# 소프트맥스 계산
print(tf.nn.softmax(c))
```

    tf.Tensor(10.0, shape=(), dtype=float32)
    tf.Tensor([1 0], shape=(2,), dtype=int64)
    tf.Tensor(
    [[2.6894143e-01 7.3105854e-01]
     [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)
    

## 형상(Shape) 정보

텐서는 모양이 있다. 사용되는 일부 용어는 다음과 같다.

*   **모양(Shape)** : 텐서의 각 차원의 길이 (요소의 수)
*   **순위(Rank)** : 텐서 축의 수. 스칼라는 rank가 0이고 벡터의 rank는 1이며 행렬의 rank는 2이다.
*   **축** 또는 **차원** : 텐서의 특정 차원
*   **크기** : 텐서의 총 항목 수. 곱 모양 벡터

**참고** : "2차원 텐서"에 대한 참조가 있을 수 있지만, rank-2 텐서는 일반적으로 2D 공간을 설명하지 않는다.

텐서 및 `tf.TensorShape` 객체에는 다음에 액세스하기 위한 편리한 속성이 있다.


```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

<tr>
  <th colspan="2"> rank-4 텐서, 모양: <code>[3, 2, 4, 5]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/shape.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/4-axis_block.png)


```python
print("모든 요소의 유형 :", rank_4_tensor.dtype)
print("차원 수 :", rank_4_tensor.ndim)
print("텐서의 모양:", rank_4_tensor.shape)
print("텐서의 축 0을 따른 요소 :", rank_4_tensor.shape[0])
print("텐서의 마지막 축에 있는 요소 :", rank_4_tensor.shape[-1])
print("총 요소 수(3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```

    모든 요소의 유형 : <dtype: 'float32'>
    차원 수 : 4
    텐서의 모양: (3, 2, 4, 5)
    텐서의 축 0을 따른 요소 : 3
    텐서의 마지막 축에 있는 요소 : 5
    총 요소 수(3*2*4*5):  120
    

축은 종종 인덱스로 참조하지만, 항상 각 축의 의미를 추적해야 한다. 축이 전역에서 로컬로 정렬되는 경우가 종종 있다. 

배치 축이 먼저 오고 그 다음에 공간 차원과 각 위치의 특성이 마지막에 온다. 이러한 방식으로 특성 벡터는 연속적인 메모리 영역이다.

<tr>
<th>일반적인 축 순서</th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/shape2.png)

## 인덱싱

### 단일 축 인덱싱

TensorFlow는 [파이썬의 목록 또는 문자열 인덱싱](https://docs.python.org/3/tutorial/introduction.html#strings)과 마찬가지로 표준 파이썬 인덱싱 규칙과 numpy 인덱싱의 기본 규칙을 따릅니다.

*   인덱스는 `0`에서 시작한다.
*   음수 인덱스는 끝에서부터 거꾸로 계산한다.
*   콜론, `:`은 슬라이스 `start:stop:step`에 사용된다.


```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
```

    [ 0  1  1  2  3  5  8 13 21 34]
    

스칼라를 사용하여 인덱싱하면 축이 제거된다.


```python
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
```

    First: 0
    Second: 1
    Last: 34
    

`:` 슬라이스를 사용하여 인덱싱하면 축이 유지된다.


```python
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
```

    Everything: [ 0  1  1  2  3  5  8 13 21 34]
    Before 4: [0 1 1 2]
    From 4 to the end: [ 3  5  8 13 21 34]
    From 2, before 7: [1 2 3 5 8]
    Every other item: [ 0  1  3  8 21]
    Reversed: [34 21 13  8  5  3  2  1  1  0]
    

### 다축 인덱싱

더 높은 rank의 텐서는 여러 인덱스를 전달하여 인덱싱된다.

단일 축의 경우와 똑같은 규칙이 각 축에 독립적으로 적용된다.


```python
print(rank_2_tensor.numpy())
```

    [[1. 2.]
     [3. 4.]
     [5. 6.]]
    

각 인덱스에 정수를 전달하면 결과는 스칼라이다..


```python
# 2-rank 텐서에서 단일 값 추출
print(rank_2_tensor[1, 1].numpy())
```

    4.0
    

정수와 슬라이스를 조합하여 인덱싱할 수 있다.


```python
# 행, 열 텐서 가져오기
print("두번째 행:", rank_2_tensor[1, :].numpy())
print("두번째 열:", rank_2_tensor[:, 1].numpy())
print("마지막 행:", rank_2_tensor[-1, :].numpy())
print("마지막 열의 첫 번째 항목:", rank_2_tensor[0, -1].numpy())
print("첫 번째 행 건너뛰기:")
print(rank_2_tensor[1:, :].numpy(), "\n")
```

    두번째 행: [3. 4.]
    두번째 열: [2. 4. 6.]
    마지막 행: [5. 6.]
    마지막 열의 첫 번째 항목: 2.0
    첫 번째 행 건너뛰기:
    [[3. 4.]
     [5. 6.]] 
    
    

Ex) 3축 텐서


```python
print(rank_3_tensor[:, :, 4])
```

    tf.Tensor(
    [[ 4  9]
     [14 19]
     [24 29]], shape=(3, 2), dtype=int32)
    

<tr>
<th colspan="2">배치에서 각 예시의 모든 위치에서 마지막 특성 선택하기</th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/index1.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/index2.png)

[텐서 슬라이싱 가이드](https://tensorflow.org/guide/tensor_slicing)에서 인덱싱을 적용하여 텐서의 개별 요소를 조작하는 방법을 알 수 있다.

## 형상(Shape) 조작하기

텐서의 모양을 바꾸는 것은 매우 유용하다.


```python
# Shape는 각 차원의 크기를 보여주는 'TensorShape' 객체를 반환한다..
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
```

    (3, 1)
    


```python
# 이 객체를 Python 목록으로 변환할 수도 있다.
print(var_x.shape.as_list())
```

    [3, 1]
    

텐서를 새로운 모양으로 바꿀 수 있다. 기본 데이터를 복제할 필요가 없으므로 재구성이 빠르고 저렴하다.


```python
# 텐서를 새로운 모양으로 변형할 수 있다.
# Note that we're passing in a list
reshaped = tf.reshape(var_x, [1, 3])
```


```python
print(var_x.shape)
print(reshaped.shape)
```

    (3, 1)
    (1, 3)
    

데이터의 레이아웃은 메모리에서 유지되고, 요청된 모양이 같은 데이터를 가리키는 새 텐서가 작성된다. 

TensorFlow는 C 스타일 "행 중심" 메모리 순서를 사용한다. 여기에서 가장 오른쪽에 있는 인덱스를 증가시키면 메모리의 단일 단계에 해당한다.


```python
print(rank_3_tensor)
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
    

텐서를 평평하게 하면 어떤 순서로 메모리에 배치되어 있는지 확인할 수 있다.


```python
# `shape` 인수에 전달된 `-1`은 모든 것을 말한다.
print(tf.reshape(rank_3_tensor, [-1]))
```

    tf.Tensor(
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29], shape=(30,), dtype=int32)
    

일반적으로 `tf.reshape`의 합리적인 용도는 인접한 축을 결합하거나 분할하는 것뿐이다(또는 `1`을 추가/제거).

3x2x5 텐서의 경우, 슬라이스가 혼합되지 않으므로 (3x2)x5 또는 3x (2x5)로 재구성하는 것이 합리적이다.

<th colspan="3">몇 가지 좋은 reshapes</th>
<tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-before.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-good1.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-good2.png)

모양을 변경하면 같은 총 요소의 수를 가진 새로운 모양에 대해 "작동"하지만, 축의 순서를 고려하지 않으면 별로 쓸모가 없다.

`tf.reshape`에서 축 교환이 작동하지 않으면, `tf.transpose`를 수행해야 한다..



```python
# 않좋은 예시

# reshape를 사용하여 축을 재정렬할 수 없습니다.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# 이건 엉망이야
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# 이것은 전혀 작동하지 않습니다.
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]]
    
     [[15 16 17 18 19]
      [20 21 22 23 24]
      [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) 
    
    InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]
    

<th colspan="3">몇 가지 잘못된 reshapes</th>
<tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-bad.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-bad4.png)
![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/reshape-bad2.png)

완전히 지정되지 않은 모양 전체에 걸쳐 실행할 수 있다. 모양에 `None`(축 길이를 알 수 없음)이 포함되거나 전체 모양이 `None`(텐서의 순위를 알 수 없음)이다.

## `DTypes`에 대한 추가 정보

`tf.Tensor`의 데이터 유형을 검사하려면, `Tensor.dtype` 속성을 사용해야 한다..

Python 객체에서 `tf.Tensor`를 만들 때 선택적으로 데이터 유형을 지정할 수 있다.

그렇지 않으면, TensorFlow는 데이터를 나타낼 수 있는 데이터 유형을 선택하면 된다. TensorFlow는 Python 정수를 `tf.int32`로, Python 부동 소수점 숫자를 `tf.float32`로 변환한다. 그렇지 않으면, TensorFlow는 NumPy가 배열로 변환할 때 사용하는 것과 같은 규칙을 사용한다..

유형별로 캐스팅할 수 있다.


```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# 이제 uint8로 변환하고 소수 자릿수를 잃습니다.
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```

    tf.Tensor([2 3 4], shape=(3,), dtype=uint8)
    

## 브로드캐스팅

브로드캐스팅은 [NumPy의 해당 특성](https://numpy.org/doc/stable/user/basics.html)에서 빌린 개념이다. 요컨대, 특정 조건에서 작은 텐서는 결합된 연산을 실행할 때 더 큰 텐서에 맞게 자동으로 "확장(streched)"된다.

가장 간단하고 가장 일반적인 경우는 스칼라에 텐서를 곱하거나 추가하려고 할 때이다. 이 경우, 스칼라는 다른 인수와 같은 모양으로 브로드캐스트됩니다. 


```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# 전부 같은 계산
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    

마찬가지로, 크기가 1인 축은 다른 인수와 일치하도록 확장할 수 있다. 두 인수 모두 같은 계산으로 확장할 수 있다.

이 경우, 3x1 행렬에 요소별로 1x4 행렬을 곱하여 3x4 행렬을 만든다. 선행 1이 선택 사항인 점에 유의하자. y의 형상은 `[4]`이다.


```python
# 전부 같은 계산
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

    tf.Tensor(
    [[1]
     [2]
     [3]], shape=(3, 1), dtype=int32) 
    
    tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 
    
    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)
    

<tr>
  <th>추가 시 브로드캐스팅: <code>[1, 4]</code>와 <code>[3, 1]</code>의 곱하기는 <code>[3,4]</code>입니다.</th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/broadcasting.png)

다음은 브로드캐스팅 없이 동일한 작업이다.


```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # 다시 말하지만, 연산자 오버로딩
```

    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)
    

대부분의 경우 브로드캐스팅은 브로드캐스트 연산으로 메모리에서 확장된 텐서를 구체화하지 않으므로 시간과 공간 효율적이다.

`tf.broadcast_to`를 사용하여 브로드캐스팅이 어떤 모습인지 알 수 있다.


```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

    tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)
    

예를 들어, `broadcast_to`는 수학적인 op와 달리 메모리를 절약하기 위해 특별한 연산을 수행하지 않는다. 여기에서 텐서를 구체화한다.

훨씬 더 복잡해질 수 있다. Jake VanderPlas의 저서 *Python Data Science Handbook*의 [해당 섹션](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)에서는 더 많은 브로드캐스팅 트릭을 보여준다(NumPy에서).

## tf.convert_to_tensor

`tf.matmul` 및 `tf.reshape`와 같은 대부분의 ops는 클래스 `tf.Tensor`의 인수를 사용한다. 그러나 위의 경우, 텐서 형상의 Python 객체가 수용됨을 알 수 있다.

전부는 아니지만 대부분의 ops는 텐서가 아닌 인수에 대해 `convert_to_tensor`를 호출한다. 변환 레지스트리가 있어 NumPy의 `ndarray`, `TensorShape` , Python 목록 및 `tf.Variable`과 같은 대부분의 객체 클래스는 모두 자동으로 변환된다.

자세한 내용은 `tf.register_tensor_conversion_function`을 참조하자. 자신만의 유형이 있으면 자동으로 텐서로 변환할 수 있다.

## 비정형(Ragged) 텐서

어떤 축을 따라 다양한 수의 요소를 가진 텐서를 "비정형(ragged)"이라고 한다다. 비정형 데이터에는 `tf.ragged.RaggedTensor`를 사용한다.

예를 들어, 비정형 텐서는 정규 텐서로 표현할 수 없다.

<tr>
  <th>`tf.RaggedTensor`, 형상: <code>[4, None]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/ragged.png)


```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
```


```python
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    ValueError: Can't convert non-rectangular Python sequence to Tensor.
    

대신 `tf.ragged.constant`를 사용하여 `tf.RaggedTensor`를 작성한다.


```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

    <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
    

`tf.RaggedTensor`의 형상에는 알 수 없는 길이의 일부 축이 포함된다.


```python
print(ragged_tensor.shape)
```

    (4, None)
    

## 문자열 텐서

`tf.string`은 `dtype`이며, 텐서에서 문자열(가변 길이의 바이트 배열)과 같은 데이터를 나타낼 수 있다.

문자열은 원자성이므로 Python 문자열과 같은 방식으로 인덱싱할 수 없다. 문자열의 길이는 텐서의 축 중의 하나가 아니다. 문자열을 조작하는 함수에 대해서는 `tf.strings`를 참조하자.

다음은 스칼라 문자열 텐서이다


```python
# 텐서는 문자열이 될 수 있습니다. 여기에서도 스칼라 문자열이 있습니다.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
```

    tf.Tensor(b'Gray wolf', shape=(), dtype=string)
    

문자열 벡터는 다음과 같다.

<tr>
  <th>문자열의 벡터, 형상: <code>[3,]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/strings.png)


```python
# 길이가 다른 세 개의 문자열 텐서가 있는 경우 괜찮다.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# 모양은 (3,)입니다. 문자열 길이는 포함되지 않는다.
print(tensor_of_strings)
```

    tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)
    

위의 출력에서 `b` 접두사는 `tf.string` dtype이 유니코드 문자열이 아니라 바이트 문자열임을 나타낸다. TensorFlow에서 유니코드 텍스트를 처리하는 자세한 내용은 [유니코드 튜토리얼](https://www.tensorflow.org/tutorials/load_data/unicode)을 참조하자.

유니코드 문자를 전달하면 UTF-8로 인코딩된다.


```python
tf.constant("🥳👍")
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>



문자열이 있는 일부 기본 함수는 tf.strings을 포함하여 tf.strings.split에서 찾을 수 있습니다.


```python
# split을 사용하여 문자열을 텐서 세트로 분할할 수 있다.
print(tf.strings.split(scalar_string_tensor, sep=" "))
```

    tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
    


```python
# 하지만 문자열의 텐서를 분할하면 'RaggedTensor'로 바뀐다.
# 각 문자열이 다른 수의 부분으로 분할될 수 있기 때문이다.

print(tf.strings.split(tensor_of_strings))
```

    <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>
    

<tr>
  <th>세 개의 분할된 문자열, 모양: <code>[3, None]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/string-split.png)

`tf.string.to_number`:


```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

    tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)
    

`tf.cast`를 사용하여 문자열 텐서를 숫자로 변환할 수는 없지만, 바이트로 변환한 다음 숫자로 변환할 수는 있다.


```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
```

    Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
    Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)
    


```python
# 또는 유니코드로 분할한 다음 디코딩.
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
```

    
    Unicode bytes: tf.Tensor(b'\xe3\x82\xa2\xe3\x83\x92\xe3\x83\xab \xf0\x9f\xa6\x86', shape=(), dtype=string)
    
    Unicode chars: tf.Tensor([b'\xe3\x82\xa2' b'\xe3\x83\x92' b'\xe3\x83\xab' b' ' b'\xf0\x9f\xa6\x86'], shape=(5,), dtype=string)
    
    Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)
    

`tf.string` dtype은 TensorFlow의 모든 원시 바이트 데이터에 사용된다. `tf.io` 모듈에는 이미지 디코딩 및 csv 구문 분석을 포함하여 데이터를 바이트로 변환하거나 바이트에서 변환하는 함수가 포함되어 있다.

## 희소 텐서

때로는 매우 넓은 임베드 공간과 같이 데이터가 희소하다. TensorFlow는 `tf.sparse.SparseTensor` 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장한다.

<tr>
  <th>`tf.SparseTensor`, 모양: <code>[3, 4]</code> </th>
</tr>

![png](https://raw.githubusercontent.com/g202jh/g202jh.github.io/master/assets/image/tensor_img/sparse.png)


```python
# 희소 텐서는 메모리-효율적인 방식으로 인덱스별로 값을 저장한다.
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# 희소 텐서를 고밀도로 변환할 수 있다.
print(tf.sparse.to_dense(sparse_tensor))
```

    SparseTensor(indices=tf.Tensor(
    [[0 0]
     [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 
    
    tf.Tensor(
    [[1 0 0 0]
     [0 0 2 0]
     [0 0 0 0]], shape=(3, 4), dtype=int32)
    
