<p align="center">
<a href=""><img src="assets/korani2.png" width="20%"></a>
</p>


# KORani

- KORani: Large Language Models for 🇰🇷 Korean and 🇺🇸 English using LLaMA 13B and Polyglot 12.8B.
- Tested which LLM is effective for 🇰🇷 Korean tasks after finetuning.
- 🤗 You can download the weights from the [Link](https://huggingface.co/KRAFTON).

## Release
This repository contains inference code for KORani models that are based on [LLaMA 13B](https://arxiv.org/abs/2302.13971v1) and [Polyglot 12.8B](https://huggingface.co/EleutherAI/polyglot-ko-12.8b).
KORani models are finetuned using [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) & [KoVicuna](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) dataset. This work is hugely influenced by [Vicuna](https://github.com/lm-sys/FastChat) project.

### Models
We offer three types of models as follows.


| Model | Base | Train dataset | Huggingface Link |
| --- | ---: | ---: | ---: |
| 1️⃣ KORani-v1-13B | Polyglot 12.8B | KoVicuna dataset | [Link 1](https://huggingface.co/KRAFTON/KORani-v1-13B) |
| 2️⃣ KORani-v2-13B | LLaMA 13B | KoVicuna dataset | [Link 2](https://huggingface.co/KRAFTON/KORani-v2-13B) |
| 3️⃣ KORani-v3-13B | LLaMA 13B | ShareGPT & KoVicuna dataset | [Link 3](https://huggingface.co/KRAFTON/KORani-v3-13B) |


### Notes
* We used LLaMA 13B from [here](https://huggingface.co/decapoda-research/llama-13b-hf).
* We extracted only the data from [Kovicuna](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) that corresponds to the first and second parts of the conversation, which are 'human' and 'GPT'.
* The model finetuning was conducted on eight A100 40GB GPUs. The code used for training is based on the [Fastchat](https://github.com/lm-sys/FastChat).

## Local Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## How to use
1. Prepare your prompt at `prompts/{task_name}.txt`
2. Run `inference.py`
```bash
python inference.py --model_path MODEL_NAME --task TASK_NAME
```
### Command
`--model_path` (str): model path for evaluation. (e.g. KRAFTON/KORani-v3-13B)  
`--task` (str): choose which task you want to evaluate. (e.g. only [QA, summarization, translation] are available in this repo.)

## Examples
You can check how to get the evaluation score in the tables from this git repository. 
https://github.com/krafton-ai/AutoEvalGPT

### 1. Question Answering (QA)

```bash
python inference.py --model_path "KRAFTON/KORani-v3-13B" --task "QA"
```

This is the prompt for QA task. You can modify it in the [QA.txt](prompts/QA.txt).

```python
PROMPT = """우리는 아래와 같은 정보를 갖고 있습니다.
---------------------
{context}
---------------------
### 주어진 정보에 따라, 질문에 답해주세요.: '{question}'
### Assistant:"""
```

For example,
```python
context = "헨리 구스타프 몰래슨(영어: Henry Gustav Molaison, 1926년 2월 26일 ~ 2008년 12월 2일)은 뇌전증을 치료하기 위해서 수술적으로 해마를 포함한 내측측두엽이 제거된 미국의 기억장애 환자 H.M으로 전부터 이미 알려져 있었다. 
그는 1957년 말부터 그가 죽을 때 까지 그는 광범위하게 연구가 되었다. 그의 경우는 뇌 기능과 기억 사이의 링크를 설명하는 이론 발달과 뇌인지심리학의 발달, 뇌의 구조와 기능이 특정 심리적 과정과 관한 이해를 목표로 하는 심리학의 한 지점에 매우 중요한 역할을 하였다. 
그가 죽기 전에, 그는 코네티컷 주 윈저 락스에 위치한 관리기관에 살면서 지속적으로 조사가 되었다. 
그의 뇌는 지금 2009년 12월 4일에 조직학적으로 슬라이스되어 미국 샌디에이고에 있다."
question = "헨리 구스타프의 뇌는 현재 어디에 위치해 있는가?"
```

| Model | Score (averaged over 5 examples) | Output Example |
| --- | :---: | ---: | 
| GPT-4 | - | 헨리 구스타프의 뇌는 현재 미국 샌디에이고에 위치해 있다. |
| GPT-3.5-turbo | 7.1 | 헨리 구스타프의 뇌는 지금 2009년 12월 4일에 조직학적으로 슬라이스되어 미국 샌디에이고에 있다. |
| Vicuna-13B | 7.2 | 헨리 구스타프의 뇌는 2009년 12월 4일에 조직학적으로 슬라이스되어 미국 샌디에이고에 있습니다. |
| KoAlpaca-13B | 2.6 | Henry Gustav Molaison의 뇌는 2008년에 사망하였습니다. 그의 뇌는 코네티컷 주 윈저 락스에 위치한 기관에 보관되어 있으며, 현재도 계속 조사중입니다. |
| KORani-v1 | 6.6 | 헨리 구스타프 몰레이슨의 뇌는 현재 미국 샌디에이고에 있습니다. 그의 뇌는 1957년 수술로 제거된 그의 내측 측두엽의 자리를 확인하기 위해 조직학적으로 절단되었습니다. 이 연구는 뇌의 구조와 기능이 특정 심리적 과정과 관련된 이해를 목표로 하는 심리학의 한 분야인 인지 신경과학에 대한 그의 공헌을 기리기 위한 것입니다." |
| KORani-v2 | 8.0 | 헨리 구스타프의 뇌는 현재 캘리포니아 샌디에이고에 있는 샌디에이고 대학교 웨스트 캠퍼스의 신경과학 실험실에 보관되어 있습니다. |
| KORani-v3 | 6.4 | 헨리 구스타프의 뇌는 현재 코네티컷 주 윈저 락스에 있는 관리 기관에 있습니다. |

### 2. Translation

```bash
python inference.py --model_path "KRAFTON/KORani-v3-13B" --task "translation"
```

This is the prompt for translation. You can modify it in the [translation.txt](prompts/translation.txt).

```python
PROMPT = """### Instruction: Translate English sentence into Korean. You may leave specific names as they are.
English: 얼마나 많은 언어를 말할 수 있니?
Korean: How many languages can you speak?#
English: 일 다 끝났어?
Korean: Did you finish your work?#
English: {source_sentence} 
Korean:"""
```
For example,
```python
source_sentence = "Hey! I have some Kissflow Legal Review requests that are blocked by Tax and Accounting, can this be looked at?"
```

| Model | Score (averaged over 5 examples) | Output Example |
| --- | :---: | ---: | 
| GPT-4 | - | 헤이! 제가 Tax and Accounting에 의해 차단된 몇 가지 Kissflow 법률 검토 요청이 있는데, 이것을 확인해 주실 수 있나요? |
| DeepL | 9.4 | 안녕하세요! 세무 및 회계에서 차단된 Kissflow 법률 검토 요청이 몇 개 있는데요, 이 요청을 살펴볼 수 있나요?  |
| GPT-3.5-turbo | 8.6 | 안녕하세요! 세무 및 회계 부서에서 차단된 몇 가지 Kissflow Legal Review 요청이 있습니다. 확인해 주실 수 있나요? |
| Vicuna-13B | 3.8 | 안녕하세요! 세금계산과 회계부서가 차단해 있는 Kissflow Legal Review 요청이 몇 개가 있습니까? 이것을 살펴보시겠습니까? |
| KoAlpaca-13B | 5.6 | 야! 세금과 회계에 막힌 키싱플로우 법적 검토 요청이 있는데, 이거 검토해줄 수 있어? |
| KORani-v1 | 7.5 | 안녕하세요! 세금과 회계로 인해 막혀 있는 키스플로우 법률 검토 요청이 몇 개 있는데, 검토해 주실 수 있나요? |
| KORani-v2 | 5.4 | 안녕하세요! 제가 Kissflow Legal Review 요청을 목격했는데, 세무 및 회계 부서에서 차단하고 있는데 이 문제를 조사해 주시겠어요? |
| KORani-v3 | 7.1 | 안녕하세요! 저는 Kissflow Legal Review 요청이 세금과 회계에 의해 차단되고 있는데, 이 문제가 살펴볼 수 있을까요? |

### 3. Summarization

```bash
python inference.py --model_path "KRAFTON/KORani-v3-13B" --task "summarization"
```

This is the prompt for summarization. You can modify it in the [summarization link](prompts/summarization.txt). Keep in mind you did not exceed the maximum length = 2048.

```python
PROMPT = """# Meeting note
{target_document}

# Summarize the meeting note into 3 Korean sentences.
### Output: 1)"""
```
For example,
```python
target_document = """# Document
전년도 대비 79명 늘어 1019명, 행정수요 대처 광양시의 공무원 정원이 크게 늘어나 행정서비스 향상이 기대된다. 
시는 행정안전부에서 발표한 2018년도 자치단체 기준인건비에서 광양시의 일반직 정원이 지난해보다 79명이 늘어난 1019명으로 산정됐다고 밝혔다. 
지난 1995년 도농통합 당시 991명이었던 광양시 공무원 정원은 IMF시절에 혹독한 구조조정을 거치면서 2002년에는 788명으로 200여명이 가까이 줄어들었으나 이번 정원 확보로 공무원 정원 1000명 시대를 맞게 됐다. 
그동안 광양시는 포스코를 중심으로 한 산업단지와 컨테이너부두, 경제자유구역, 택지 개발, 다양한 복지 정책 등 시민의 삶의 질 향상을 위한 행정수요가 날로 증폭하는 데에 비해 한정된 공무원 정원으로 많은 어려움을 겪어 왔었다. 
시의 이번 정원 충원은 지금까지 격무에 시달려온 공무원들에게 더욱 열심히 일할 수 있도록 활력을 불어넣으면서 지역과 나라 발전을 뒷받침 하는 경제도시로서의 위상을 드높이는 시너지 효과로 이어질 것으로 기대된다. 
한편, 광양시는 한시기구인‘기업유치추진단’이 2017년 연말로 폐지됨에 따라 전라남도로부터 4급 상설기구인‘산단녹지관리센터’를 승인 받아 올해 1월부터 운영하게 됨으로써 도시 개발에 한층 탄력을 받을 것으로 전망하고 있다. 
정홍기 총무과장은“참으로 어렵게 가져온 공무원 정원과 4급 상설기구는 격무에 시달려온 직원들의 숨통을 트이게 해 15만 시민을 위한 행정서비스 질을 향상시킬 것으로 기대된다”며“이달 안에 정원 조례를 개정하고, 직무분석을 통해 부서별 정원을 조정할 계획이다”고 말했다.

한국어 3문장으로 위의 Document를 요약해줘.
### Assistant: 1)"""
```

| Model | Score (averaged over 5 examples) | Output Example |
| --- | :---: | ---: | 
| GPT-4 | - | 1) 광양시의 공무원 정원이 전년도 대비 79명 늘어난 1019명으로 증가했다. 2) 이로 인해 행정서비스 향상이 기대되며, 지역과 나라 발전을 뒷받침하는 경제도시로서의 위상이 상승할 것으로 전망된다. 3) 한편, 광양시는 전라남도로부터 4급 상설기구인 '산단녹지관리센터'를 승인받아 도시 개발에 탄력을 받을 것으로 예상된다. |
| GPT-3.5-turbo | 8.5 | 1) 광양시 공무원 정원이 79명 늘어 1019명으로 늘어나 행정서비스 향상이 기대된다고 밝혔다.2) 광양시는 이번 정원 충원으로 지역과 나라 발전을 뒷받침 하는 경제도시 위상을 드높일 것으로 기대된다. 3) 정원 조례를 개정하고, 직무분석을 통해 부서별 정원을 조정할 계획이다. |
| Vicuna-13B | 7.4 | 1) 2018년도 광양시의 공무원 정원 1019명, 지난해 79명 증가. 2) 이번 정원 충원은 지금까지 격무에 시달려온 공무원들에게 더욱 열심히 일할 수 있도록 활력을 불어넣으면서 지역과 나라 발전을 뒷받침 하는 경제도시로서의 위상을 드높이는 시너지 효과로 이어질 것으로 기대된다. 3) 광양시는 전라남도로부터 4급 상설기구인‘산단녹지관리센터’를 승인 받아 운영하게 됨으로써 도시 개발에 한층 탄력을 받을 것으로 전망하고 있다. |
| KoAlpaca-13B | 5.7 | 1) 광양시는 작년보다 79명 늘어난 1019명의 공무원을 둘 수 있게 되었습니다. 2) 이는 행정안전부의 지침에 따른 것입니다. 3) 이로써 광양시는 한정된 예산으로 인해 발생했던 공무원들의 초과 근무 수당 지급 등의 문제를 해결 할 수 있게 되었습니다. |
| KORani-v1 | 7.0 | 1) 광양시는 작년보다 79명이 늘어난 1019명의 일반직 공무원 정원을 행안부로부터 승인받았습니다. 2) 이는 광양시의 행정 수요 증가를 수용하기 위한 것입니다. 3) 이 정원 증가는 광양시 공무원들에게 더 많은 기회를 제공할 것입니다. |
| KORani-v2 | 6.2 | 1)광양시의 공무원 정원은 이번 정원 확보로 1000명 주당 공무원 정원을 달성했습니다. 2) 이러한 충원은 지금까지 격무에 시달려온 공무원들에게 더욱 열심히 일할 수 있도록 활력을 불어넣으면서 지역사회와 국가의 발전을 지원하는 시너지 효과로 이어질 것으로 기대됩니다. 3) 광양시는 '기업유치추진단'을 폐지하고 '산단녹지관리센터'를 승인했으며, 이는 도시 개발에 한층 탄력을 받을 것으로 기대됩니다. |
| KORani-v3 | 6.9 | 1)광양시의 공무원 정원은 지난해보다 79명이 늘어난 1019명으로 산정되었습니다.2) 이러한 충원은 지역사회와 국가 발전을 지원하는 행정서비스의 향상으로 이어질 것으로 기대됩니다.3) 시는 '이달 내에 정원 조례를 개정하고 직무분석을 통해 부서별로 정원을 조정할 계획'이며 행정서비스 향상을 위해 노력할 것이라고 밝혔습니다. |

## Evaluation
We tested model performance using GPT-4, and the code and results of the test can be found through the [AutoEvalGPT](https://github.com/krafton-ai/AutoEvalGPT).

## Limitations
The Korean performance of our models is not as good as the English performance of [Vicuna](https://github.com/lm-sys/FastChat). We believe this is due to the not enough quality of foundation models in the Korean tasks (compared to Llama in English tasks) and the dataset quality, which is primarily translational. We will continue to update the new versions of the Korani models as soon as we achieve better results.

## License
Our github repo and models are intended for research purpose, non-commercial use only, subject to the model License of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us If you find any potential violation.
The code is released under the Apache License 2.0.
