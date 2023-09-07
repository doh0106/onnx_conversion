# onnx_conversion

### ONNX란?
ONNX는 쉽게 말하면 다양한 framework(Pytorch, TensorFlow 등)로 학습된 머신러닝 모델들을 나타내는 하나의 표준 형식입니다. 즉 모델의 저장 형식(확장자)이라고 봐도 됩니다. 즉 같은 모델에 대해서 pytorch의 경우 model.pt, tensorflow의 경우 model.pb, ONNX의 경우 model.onnx인 것 입니니다. 

### 왜 ONNX를 쓸까?
저는 개인적으로 처음에 딥러닝을 tensorflow로 배웠습니다. 그래서 당시 reference를 찾으면 대부분 pytorch 관련 자료만 나와서 사용할 수 없어서 실망했던 적이 많습니다. 
바로 이런 점을 해결할 수 있는 것이 ONNX입니다. 다양한 framework들에서 학습되고 구성된 모델을 다른 framework에서 사용하는것이 힘듭니다. 
하지만 ONNX는 프레임워크에 상관없이 모델을 저장하고 전달할 수 있게 해줍니다.
예를 들어, PyTorch에서 학습한 모델을 ONNX 포맷으로 변환하고, 이를 다시 TensorFlow나 ONNX Runtime 등에서 실행할 수 있습니다.


> 정의 : 머신 러닝 모델의 표준 포맷
용도 : 다른 framework간의 모델 호환성 부여

### ONNX 변환방법
저는 허깅페이스에서 쉽게 받을 수 있는 bert 기반의 모델을 ONNX로 바꿔보았습니다. 크게 3가지의 방법(high-level, mid-levl, low-level)이 있습니다. high-level의 방법일 수록 변환방법이 간편하지만 세세한 부분을 설정할 수 없어 오히려 불편할 수도 있으므로 high, low 방법에 대해 설명하겠습니다. 

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

# 변경할 onnx모델의 형식 지정
torch.onnx.export(
    model,							# model 위치
    tuple(inputs.values()),			# input
    f="/root/work/temp.onnx",		# onnx모델 저장 경로
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":{0: "batch_size", 1: "sequence"},
        "attention_mask":{0: "batch_size", 1: "sequence"},
        "token_type_ids":{0: "batch_size", 1: "sequence"},
        "logits":{0:'batch_size'},
                  },
    do_constant_folding=True, 
    opset_version=13,
)
```

