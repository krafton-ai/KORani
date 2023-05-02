# KORani

- KORani: Large Language Models for 🇰🇷 Korean and 🇺🇸 English using LLaMA 13B and Polyglot 12.8B.
- Tested which LLM is effective for 🇰🇷 Korean tasks after finetuning.
- 🤗 You can download the weights from the [Link](https://huggingface.co/KRAFTON).

## Release
<p align="center">
<a href=""><img src="assets/KORani.png" width="20%"></a>
</p>
 
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

## Inference
### Command <br>
    `--model_path` (str): model path for evaluation. (e.g. KRAFTON/KORani-v3-13B) <br>
    `--task` (str): choose which task you want to evaluate. (e.g. only [QA, summarization, translation] are available in this repo.) <br>


### Question Answering

    ```bash
    python inference.py \
        --model_path "KRAFTON/KORani-v3-13B"
        --task "QA"
    ```

    Below is the prompt used to generate the answer. You can modify it in the [QA link](prompts/QA.txt).

    ```python
    PROMPT = """\
    우리는 아래와 같은 정보를 갖고 있습니다.
    ---------------------
    {context}
    ---------------------
    ### 주어진 정보에 따라, 질문에 답해주세요.: '{question}'
    ### Assistant:"""
    ```

### QA sample result
```python
context = ""
question = ""
```

| Model | Target |
| --- | ---: |
| KORani-v1-13B |  |
| KORani-v2-13B |  |
| KORani-v3-13B |  |
| Vicuna 13B |  |
| Koalpaca-13B |  |
| ChatGPT 3.5 |  |
| GPT4 |  |

### Translation

    ```bash
    python inference.py \
        --model_path "KRAFTON/KORani-v3-13B"
        --task "translation"
    ```

    Below is the prompt used to generate the answer. You can modify it in the [translation link](prompts/translation.txt). If you use few-shot in translation, the performance can improve.

    ```python
    PROMPT = """\
    ### Instruction: Translate English sentence into Korean. You may leave specific names as they are.
    English: 얼마나 많은 언어를 말할 수 있니?
    Korean: How many languages can you speak?#
    English: 일 다 끝났어?
    Korean: Did you finish your work?#
    English: {source_sentence} 
    Korean:"""
    ```

### Translation sample result
```python
source_sentence = "Hey! I have some Kissflow Legal Review requests that are blocked by Tax and Accounting, can this be looked at?"
```

| Model | Target |
| --- | ---: |
| KORani-v1-13B | "안녕하세요! 세금과 회계로 인해 막혀 있는 키스플로우 법률 검토 요청이 몇 개 있는데, 검토해 주실 수 있나요?" |
| KORani-v2-13B | "안녕하세요! 제가 Kissflow Legal Review 요청을 목격했는데, 세무 및 회계 부서에서 차단하고 있는데 이 문제를 조사해 주시겠어요?" |
| KORani-v3-13B | "안녕하세요! 저는 Kissflow Legal Review 요청이 세금과 회계에 의해 차단되고 있는데, 이 문제가 살펴볼 수 있을까요?" |
| Vicuna 13B | "안녕하세요! 세금계산과 회계부서가 차단해 있는 Kissflow Legal Review 요청이 몇 개가 있습니까? 이것을 살펴보시겠습니까?" |
| Koalpaca-13B | "야! 세금과 회계에 막힌 키싱플로우 법적 검토 요청이 있는데, 이거 검토해줄 수 있어?" |
| DeepL | "안녕하세요! 세무 및 회계에서 차단된 Kissflow 법률 검토 요청이 몇 개 있는데요, 이 요청을 살펴볼 수 있나요?" |
| ChatGPT 3.5 | "안녕하세요! 세무 및 회계 부서에서 차단된 몇 가지 Kissflow Legal Review 요청이 있습니다. 확인해 주실 수 있나요?" |
| GPT4 | "헤이! 제가 Tax and Accounting에 의해 차단된 몇 가지 Kissflow 법률 검토 요청이 있는데, 이것을 확인해 주실 수 있나요?" |


### Summarization

    ```bash
    python inference.py \
        --model_path "KRAFTON/KORani-v3-13B"
        --task "summarization"
    ```

    Below is the prompt used to generate the answer. You can modify it in the [summarization link](prompts/summarization.txt). It does not work for a max length of over 2048.

    ```python
    PROMPT = """\
    # Meeting note
    {target_document}

    # Summarize the meeting note into 3 Korean sentences.
    ### Output: 1)"""
    ```

### Summarization sample result

```python
target_document = ""
```

| Model | Target |
| --- | ---: |
| KORani-v1-13B |  |
| KORani-v2-13B |  |
| KORani-v3-13B |  |
| Vicuna 13B |  |
| Koalpaca-13B |  |
| ChatGPT 3.5 |  |
| GPT4 |  |

## Evaluation
We tested model performance using GPT-4, and the code and results of the test can be found through the [AutoEvalGPT](https://github.com/krafton-ai/AutoEvalGPT).

## Limitations
The Korean performance of our models is not as good as the English performance of [Vicuna](https://github.com/lm-sys/FastChat). We believe this is due to the not enough quality of foundation models in the Korean tasks (compared to Llama in English tasks) and the dataset quality, which is primarily translational. We will continue to update the new versions of the Korani models as soon as we achieve better results.

## License
Our github repo and models are intended for research purpose, non-commercial use only, subject to the model License of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us If you find any potential violation.
The code is released under the Apache License 2.0.
