# LSTM 모델

## Recurrent Neural Network

- 전통적인 neural network의 단점 : 이전에 일어난 사건을 바탕으로 나중에 일어나는 사건을 생각하지 못한다.
- RNN은 이 문제를 해결한다.
    - 스스로를 반복하면서 이전 단계에서 얻은 정보가 지속되도록 한다.

![alt text](image.png)

- A : RNN의 한 덩어리. input `Xt`를 받아서 `ht`를 내보낸다. A를 둘러싼 반복은 다음 단계에서의 network가 이전 단계의 정보를 받는다는 것을 보여줌.
- 하나의 network를 계속 복사해서 순서대로 정보를 전달하는 network  
=> sequence나 list와 같은 데이터를 다루기에 최적화된 구조

## LSTM (Long Short-Term Memory Network)


- 현재 시점의 무언가를 얻기 위해서 멀지 않은 최근 정보만 필요로 할 때가 있음.
- 하지만, 반대로 더 많은 문맥을 필요로 하는 경우도 있음.  
=> long-term dependencies를 이론 상 완벽하게 다룰 수 있다고 한다.
- 긴 의존 기간의 문제를 피하기 위해 명시적으로 설계됨.

![alt text](image-1.png)
![alt text](image-2.png)

- vector trancfer : 한 노드의 output을 다른 노드의 input으로 vector 전체를 보내는 흐름
- Pointwise Operation : ex) vector 합
- Neural Network Layer : 학습된 뉴럴네트워크 층

### LSTM의 핵심 아이디어

- cell state : 모듈 그림에서 수평으로 그어진 윗 선 (like 컨베이어 벨트)
    - 작은 linear interaction만을 적용시키면서 전체 체인을 계속 구동시킨다.
    - 정보가 전혀 바뀌지 않고 그대로 흐르게만 하는 것을 매우 쉽게 할 수 있음.

- gate : cell state에 뭔가를 더하거나 없앨 수 있는 능력
    - 정보가 전달될 수 있는 추가적인 방법
    - sigmoid layer와 pointwise 곱셈으로 이루어짐.

- sigmoid layer : 0과 1 사이의 숫자
    - 각 컴포넌트가 얼마나 정보를 전달해야 하는지에 대한 척도
    - 0 : 아무것도 넘기지 말라
    - 1 : 모든 것을 넘겨드려라

=> LSTM은 3개의 gate를 가지고 있고, 이 문들은 cell state를 보호하고 제어함.

### 1단계

- cell state로부터 어떤 정보를 버릴 것인가?
    - sigmoid layer에 의해 결정

=> forget gate layer
- `ht-1`과 `xt`를 받아서 0과 1 사이의 값을 `Ct-1`에게 준다.
- 그 값이 1이면 "모든 정보를 보존해라", 0이면 "죄다 갖다버려라"

### 2단계

- 앞으로 들어오는 새로운 정보 중 어떤 것을 cell state에 저장할 것인가?
    - `input gate layer` : 어떤 값을 업데이트할지 정함.
    - `tanh layer` : 새로운 후보값들인 `tildeCt` vector 생성, cell state에 더할 준비

=> 1, 2단계에서 나온 정보를 합쳐서 state를 업데이트할 재료를 만들게 됨.

### 3단계

- 과거 state인 `Ct-1`를 업데이트해서 새로운 cell state인 `Ct`를 생성
    - 이전 state에 `ft`곱해서 가장 첫 단계에서 잊어버리기로 정했던 것들을 진짜로 잊어버림.
    - `it`*`tildeC_t`를 더함.
        - 두 번째 단계에서 업데이트하기로 한 값을 얼마나 업데이트할 지 정한 만큼 scale한 값이 됨.

### 4단계

- 무엇을 output으로 내보낼지 정하기
    - output : cell state를 바탕으로 필터된 값
    - sigmoid layer에 input 데이터를 태워서 cell state의 어느 부분을 output으로 내보낼지 선정.
    - cell state를 tanh layer에 태워서 -1과 1 사이의 값을 받은 후, sigmoid gate의 output과 곱해줌.