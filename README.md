# onnx_conversion

### ONNX란?
ONNX는 쉽게 말하면 다양한 framework(Pytorch, TensorFlow 등)로 학습된 머신러닝 모델들을 나타내는 하나의 표준 형식입니다. 즉 모델의 저장 형식(확장자)이라고 봐도 됩니다. 즉 같은 모델에 대해서 pytorch의 경우 model.pt, tensorflow의 경우 model.pb, ONNX의 경우 model.onnx인 것 입니니다. 

<br>

### 왜 ONNX를 쓸까?
저는 개인적으로 처음에 딥러닝을 tensorflow로 배웠습니다. 그래서 당시 reference를 찾으면 대부분 pytorch 관련 자료만 나와서 사용할 수 없어서 실망했던 적이 많습니다. 
바로 이런 점을 해결할 수 있는 것이 ONNX입니다. 다양한 framework들에서 학습되고 구성된 모델을 다른 framework에서 사용하는것이 힘듭니다. 
하지만 ONNX는 프레임워크에 상관없이 모델을 저장하고 전달할 수 있게 해줍니다.
예를 들어, PyTorch에서 학습한 모델을 ONNX 포맷으로 변환하고, 이를 다시 TensorFlow나 ONNX Runtime 등에서 실행할 수 있습니다.


> 정의 : 머신 러닝 모델의 표준 포맷
용도 : 다른 framework간의 모델 호환성 부여

<br>

### ONNX 변환방법
저는 허깅페이스에서 쉽게 받을 수 있는 bert 기반의 모델을 ONNX로 바꿔보았습니다. 크게 3가지의 방법(high-level, mid-levl, low-level)이 있습니다. high-level의 방법일 수록 변환방법이 간편하지만 세세한 부분을 설정할 수 없어 오히려 불편할 수도 있으므로 high, low 방법에 대해 설명하겠습니다. 

<br>

#### Optimum-cli 이용 (high-level)
transformers 모델에 한정해서는 huggingface에서 제공하는 optimum command line interface를 이용해 매우 간단하게 모델을 onnx로 바꿀 수 있습니다. 
우선 optimum-cli 라이브러리를 설치해야 하는데 다음과 같은 명령어를 사용합니다.

```bash
pip install optimum[exporters]
```

이후 다음과 같은 명령어를 통해 model을 onnx 형식으로 바꿀 수 있습니다. 
단 모델이 저장된 경로를 지정해 주어야 하므로 huggingface의 모델 카드에서 직접 파일을 받아서 저장해두었습니다.
그리고 사용하는 모델의 종류에 따라 --task를 다르게 설정해 주시면 됩니다. 
```bash
optimum-cli export onnx -m {path/to/torch/modeldir/} --task text-classification {path/to/outputdir/}
```
<br>

#### torch.onnx 이용 (low-level)
저는 torch 모델을 변환할 것이기 때문에 torch.onnx를 이용합니다. tensorflow의 경우 `tf2onnx`를 사용하면 된다고 합니다.


```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model
model_path = "distilbert-base-uncased-finetuned-sst-2-english"     # 허깅 페이스의 모델 이름, 자신의 모델이 저장된 경로
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

dummy_model_input = tokenizer("This is a sample", return_tensors="pt")  # 모델의 input - output 체크를 위해 input을 넣어주어야 합니다.
print(dummy_model_input.keys())        # 아래의 input_names에 입력할 이름 확인

# 변경할 onnx모델의 형식 지정
torch.onnx.export(
    model,							                # 이전 코드에서 생성된 model 
    tuple(dummy_model_input.values()),			    # input의 값들
    f="/root/onnx/model.onnx",		                # onnx모델 저장 경로
    input_names=['input_ids', 'attention_mask'],    # input들의 이름
    output_names=["logits"],                        # output의 이름
    dynamic_axes={                                  # input, output의 shape중에 가변적인 부분의 dim이 존재할 경우 
        "input_ids":{0: "batch_size", 1: "sequence"},    # 해당 부분의 axis와 description을 작성
        "attention_mask":{0: "batch_size", 1: "sequence"},
        "logits":{0:'batch_size'},
                  },
    do_constant_folding=True, 
    opset_version=13,
)
```
bert 모델의 경우 `[batch_size, seq_len]`으로 input이 들어가게 되고 `[batch_size, 2]` output은 이와 같이 나옵니다. 
2는 긍부정을 판단하는 모델이기 때문에 고정적이지만, batch_size, seq_len은 가변적입니다. 그래서 위에서 dynamic_axes에서 정의해 주어야 합니다.
do_constant_folding과 opset_version은 추후에 설명 업데이트 필요

<br>

#### ONNX 모델 실행

```python
import onnx 
import onnxruntime as ort 

onnx_path = "/root/onnx/model.onnx"
onnx_model = onnx.load(onnx_path)

# onnx inference위해 session 생성
sess = ort.InferenceSession(onnx_path)

# onnx에 넣어줄 input 형식 만들어 주기
onnx_input = {}
for input in sess.get_inputs():
    input_name = input.name
    input_data = dummy_model_input[input_name].numpy()        # 여기서 dummy_model_input은 위에 모델 변화하는 코드에서 생성되었습니다.
    onnx_input[input_name]=input_data
onnx_output = sess.run(None, onnx_input)                      # onnx의 infer 결과

```
이러한 과정을 통하여 onnx모델에 대하여 output을 생성해 볼 수 있습니다.

<br>

#### ONNX모델과 pytorch모델의 결과값 비고
```python
import torch
with torch.no_grad():
    model.to('cuda') 
    outputs = model(**dummy_model_input.to('cuda'))
    model.to('cpu')

import numpy as np 
np.allclose(outputs.logits.cpu(), onnx_output)

```
자신의 모델이 onnx형태로 바뀌었고 결과값이 나온다지만 정말로 원본 모델과 같은 값을 inference하는지 의심이 갈 수도 있습니다.
그럴때, 위와 같이 `outputs`은 원본 모델의 결과값이고 `onnx_output`은 onnx모델의 결과값을 np.allclose()를 통하여 둘에 값들이 정말 같은지(오차를 어느정도 허용할지 `np.allclose()`안의 파라미터 값으로 설정가능)확인하실 수 있습니다. `np.allclose()`의 값이 True이면 모든 값들의 차이가 오차범위안이라는 것입니다.




