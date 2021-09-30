# EfficientNet 논문 리뷰 

[EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)논문을 읽고 그 내용을 여기에 요약&기록합니다.

##  1. Abstract

기존의 CNN은 고정된 자원에 제한되어 개발이 되었고, 이용할 수 있는 자원이 늘어남에 따라 그에 상응하는 더 높은 정확도를 내기위해 확장되었다. 
**이 논문에서는 이러한 모델의 확장(Model scaling)에 초점을 맞춰 체계적으로 네트워크의 깊이, 너미 및 해상도의 균형을 연구하여 모델의 성능향상을 증명한다.**  

이 논문에는 model의 depth/width/resolution에 대한 ***compound coefficient***를 사용하여 모든 차원을 균일하게 확장하는 새로운 scaling 방법을 제안한다. 

이러한 증명은 MobileNet과 ResNet의 확장으로 이루어 진다. 

또한, **NAS**(Neural Architecture Search - 강화학습 기반 최적 network 탐색 방법)를 이용하여 새로운 baseline network를 설계하고 이 모델을 scaling하여 기존의 CNN모델보다 더 좋은 성능의 **EfficientNet**을 설명한다.
EfficientNet-B7은 ImageNet에서 84.3%의 top-1 accuracy 성능으로 SOTA를 달성하였다.



## 2. Introduction

 이 논문에서 언급하는 scaling 이라는 것은 모델의 depth, wide, resolution의 크기를 조절하여 모델의 성능을 높이는 방법을 의미한다. 
지금까지의 CNN의 발전 방향을 보면 이러한 scaling이 많이 쓰였다. 

![Screenshot from 2021-09-27 15-15-39](https://user-images.githubusercontent.com/77032455/134854169-9537da29-49f4-4686-971e-9a5edf8d042d.png)

예를 들어 ResNet의 경우 layer를 18에서 200으로 증가시킴으로써 모델의 성능을 향상시킬 수 있었다. 최근에는 Gpipe라는 모델 또한 baseline 모델을 4배 확장시킴으로써 84.3%의 정확도를 낼 수 있었다. 

이러한 식으로 모델을 scale을 확장함으로써 성능을 높이는 것이 가능했지만 이러한 scaling방법에 대한 근거는 명확하지 않고 여러 방법이 존재한다. 

이전의 연구에서는 depth, wide, resolution 의 3차원에 대해서 각각 1개의 차원을 확장함으로써 모델의 성능을 높이는 것이 대부분이었다. 간혹 2,3개의 차원에 대해서 확장을 시도한다고 해도 어디까지나 사람이 임의로 설정하는 값으로 확장이 되며 이는 수동조정으로 인한 인력이 많이 소모된다. 

**따라서 이 논문에서는 depth, wide, resolution의 3차원을 조정함으로써 더욱 좋은 성능을 낼수 있는 이론을 찾는다.** 

여러 경험적인 실험을 바탕으로 네트워크의 폭, 깊이, 해상도 3차원에 대해서 균형을 맞추는 것이 중요하며 이러한 균형은 **각각의 크기를 일정한 비율로 확장**하기만 하면 달성할 수 있다. 

이 논문에서는  fixed scaling coefficients를 사용하여 3차원의 값을 균일한 비율로 스케일링한다. 

예를 들어, 2^N배 모델을 확장하고 싶다면 depth, wide, resolution 또한 ^N배 해서 적용한다. 
$$
예를\,들어, \,2^N배 \,모델을\, 확장하고 \,싶다면 \,depth(\alpha), \,wide(\beta),\, resolution(\gamma) \,에 \,대해서도\, 이를 \,적용하여 \,\alpha^N,\, \beta^N, \,\gamma^N 배 \,하여\, 3개의\,차원에 \,대해서 \,같은\, 비율을 \,적용해\, 준다. 
$$
여기서 alpha, beta, gamma를 찾는 것은 작은 grid search를 이용하여 찾도록 한다. 





![Screenshot from 2021-09-27 15-50-34](https://user-images.githubusercontent.com/77032455/134858403-65379c6e-3491-4273-8fe4-e3d6c20ef891.png)

위 이미지는 depth, width, resolution을 각각 scaling했을 경우와 compound scaling을 했을 때의 모델의 변화를 시각적으로 보여준다. 

이 논문에서 주장하는 모델의 설계방법을 요약해 보면 우선 작은 크기의 baseline model을 만든다. 이 baseline model을 만들때는 **NAS**를 이용하여 좋은 성능의 모델을 만든다. 그 이유는 Model scaling으로 인한 성능향상이 baseline model의 성능에 매우 의존적이기 때문이다.
이러한 baseline model에 대해서 compound scaling을 적용하면 EfficientNet이 되는 것이다.  


## 3. Compound Model Scaling

이제 이 논문에서 사용한 compund model scaling에 대해서 알아보자. 

### 1) Problem Formulation

![Screenshot from 2021-09-28 01-49-21](https://user-images.githubusercontent.com/77032455/134951798-380f5912-c686-4646-8533-ec93bbd56da2.png)

이 논문에서는 이미 정의된 (baseline model에서) 매개변수인 F,L, H, W, C를 제외한 w,d,r을 찾는다. w는 width에 대한 계수이며 C에 곱해진다. d는 depth에 대한 계수이며 L에 곱해진다. 마지막으로 r은 해상도에 대한 계수이므로 H,W에 곱해진다. 



### 2) Scaling Dimensions

![Screenshot from 2021-09-28 01-56-52](https://user-images.githubusercontent.com/77032455/134952937-a25297b6-92e5-4bfd-9bb4-7dd1e99cfc29.png)

#### (1) Depth(d) 

depth 방향으로의 scaling은 CNN에서 사용하는 가장 일반적인 scaling방법이다. 하지만 경험적 연구를 통해서 depth방향으로의 scaling은 깊이가 깊어졌을 경우 한계를 나타내는 것을 확인할 수 있었다. 예를 들어 ResNet -101은 ResNet-1000과 비슷한 정확도를 가진다. 가운데 이미지에서 그 결과를 확인할 수 있다. 

#### (2) Width(w)

width를 이용한 scaling기법은 보통 작은 모델에 적용되는 것이 일반적이다. 
width를 이용한 scaling을 하면 더 세분화된 feature들을 포착할 수 있는 경향이 있지만 높은 수준의 feature를 포착하는데 한계가 있다. 왼쪽 이미지에서 그 결과를 확인할 수 있다. 

#### (3) Resolution(r)

resolution을 통한 scaling기법은 단순히 input데이터로 해상도가 높은 이미지를 넣어주는 것을 의미하는데 해상도가 높을수록 더 미세한 패턴을 캡처할 수 있다. 하지만 resolution도 depth와 width 의 경우와 마찬가지로 일정 수준 이상으로 높아지게 되면 그에 따른 모델의 정확도 향상이 더뎌지게 된다. 오른쪽 이미지에서 확인할 수 있다. 



> **이러한 결과를 바탕으로 깊이, 너비, 해상도 모든 차원에 대해서 scaling up을 실행할 경우 모델의 성능이 증가하지만 지나치게 커질 경우 모델 성능의 증가량은 감소하는 것을 알 수 있다.** 



### 3) Compound Scaling

고해상도의 이미지가 input으로 들어올 경우 receptive fields가 커지고 넓은 면적에 대한 high level feature를 잘 포착하기 위해서는 depth가 증가하여야 하고 width를 증가시켜 이미지에 대한 더욱 세분화된 패턴을 캡처해야 한다는 것을 알 수 있다. 
따라서 이 3개의 차원에 대해서 scaling 치수를 조정하고 균형을 맞춰야 한다. 



![Screenshot from 2021-09-28 02-26-37](https://user-images.githubusercontent.com/77032455/134956670-d8ddc1e2-a584-44db-bfc5-4b6f46d282ed.png)

위 표에서는 d,w,r의 균형을 맞춰는 것에 대한 중요성을 나타낸다. d와 r을 고정하고 w만 키웠을 경우 기본모델에서는 낮은 정확도에서 한계를 보이지만 d와 r이 커진 이후에 w도 점차 증가시키면 높은 정확도를 내는 것을 확인할 수 있다. 



> **정확성과 효율성을 높이려면 ConvNet 확장 중에 네트워크 폭, 깊이 및 해상도의 모든 차원의 균형을 맞추는 것이 중요하다.**



이전의 연구에서는 d와 w의 균형을 임의로 조정하여 모델의 성능을 높이려는 시도가 있었지만 이러한 시도들은 어디까지나 임의로 시도된 것이기 때문에 한계가 있었고, 이 논문에서는 원칙적인 방법을 통해 3가지 차원에 대한 scaling 방법을 제시한다. 



![Screenshot from 2021-09-28 02-35-01](https://user-images.githubusercontent.com/77032455/134957753-829b44db-9ee0-4111-9091-162787bee0eb.png)



d, w, r을 정할 때, 위와 같이 alpha, beta, gamma 로 표현할 수 있고 이 값들은 작은 grid search를 통해 찾아진다. 이때 위에 붙은 phi는 사용자가 임의로 지정하는 계수이며 이는 모델을 얼마나 scaling할지를 나타낸다. 

모델의 scale을 무한대로 늘릴수는 없기 때문에 이 논문에서는 한가지 제한사항을 두었다. 

 **s.t** 부분에 기입되어 있는것이 그것인데 alpha는 scaling하면 그대로 연산량이 비례하여 증가하지만 beta와 gamma의 경우 scaling할 경우 제곱에 비례하여 연산량이 증가한다. 그들의 곱을 2가 되도록 제한하여 현실적인 모델의 scaling 한도를 둔 것이다. 



## 4. EfficientNet Architecture



![Screenshot from 2021-09-28 03-18-31](https://user-images.githubusercontent.com/77032455/134963925-ca64351b-a80f-4aae-af12-eddf8461540a.png)



EfficientNet의 기본구조인 EfficientNet-B0모델은 MBConv(MobileNetV1)과 SE(squeeze-and-excitation)을 기반으로 만들어 졌다. 

baseline-model로써 만들어진 EfficientNet-B0를 기반으로 scaling compound scaling을 하여 점차 성능을 높힌다. 

그러한 방법에는 아래의 2단계의 절차를 수행한다. 



![Screenshot from 2021-09-28 03-12-35](https://user-images.githubusercontent.com/77032455/134963107-aad63ae8-ba4d-4d89-b5c8-9dffc64b6dce.png)

- **STEP 1 **
  phi를 1로 고정시키고 2배의 자원이 할당된다고 가정하여 grid search를 사용하여 최적의 alpha, beta, gamma를 찾는다. EfficientNet-B0에 대해서 찾아진 값들은 각각 1.2, 1.1, 1.15로 찾아졌다. 
- **STEP 2**
  찾은 alpha, beta, gamma를 고정시키고 phi를 변경하여 scaling 된 모델을 얻는다. 



## 5. Experiment

이제 위 방법으로 만들어진 EfficientNet을 기존의 다른 모델들과 비교해 보자.



![Screenshot from 2021-09-28 03-24-25](https://user-images.githubusercontent.com/77032455/134964625-fad45a28-ed8d-4162-b6d8-02273265e7ff.png)

이 표에서 확인할 수 있듯이 비슷한 성능을 가진 모델에 비해 parameter의 수와 계산량이 현저히 적은것을 확인할 수 있다. 

그래프로 도식화 해봤을 경우는 다음과 같다. 

![Screenshot from 2021-09-28 03-26-41](https://user-images.githubusercontent.com/77032455/134964926-80f7deb9-f4ac-4417-bf61-05c0d11568f9.png)


