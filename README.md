# SS_height
삼성 SEM image depth map estimation

주어진 데이터가 unlabeled 실제 데이터와 labled simulation data 였음.

background 색이 140 150 160 170 으로 동일하게 있던것을 잘 catch해서 진행했어야 하는데 늦게 알아냄

뒤늦게 알아내고 embedding layer 적용해서 배경의 분류 결과에 따라 adaptive 하게 depth map 생성하려고 하는데 embedding layer input을 one-hot으로 넣는 실수를함

보통 인터넷이나 공부하는 자료들의 설명상으로는 class 를 one-hot 형태로 하여 embedding layer의 input으로 사용하는듯이 표현되어 있느나 pytorch의 경우 one-hot이 아닌 

단순 class 값 ex) 2, 5 ... 의 값을 넣어야 하는데 one-hot으로 하여 학습이 엉망으로 되었음...

구현시 처음 사용하거나 해깔리는 항목있으면 torch.ones(....) 로 원하는 형태로 만들어서 layer연산 돌아가는 것만 확인해도 실수를 줄일 수 있을것으로 생각됨

최종 private score 44등/139
