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

### Sample result
```python
source sentence = "Israel on Thursday expressed disappointment at the decision of U.S. President Donald Trump to sign a waiver to delay relocating the U.S. Embassy in Israel to Jerusalem from Tel Aviv but said it hoped a move could take place later."
```

    | Model | Target |
    | --- | ---: |
    | KORani-v1-13B | "이스라엘은 목요일 미국 대통령 도널드 트럼프가 예루살렘에 있는 미국 대사관을 텔아비브에서 예루살렘으로 이전하는 것을 연기하기로 한 결정에 대해 실망을 표명했지만, 향후 이전이 이루어질 것으로 기대한다고 말했습니다." |
    | KORani-v2-13B | "이스라엘은 요일 미국 대통령 도널드 트럼프가 제안한 것으로 보이는 지중해에서 유대인 주민들을 위해 미국 영사관을 예정된 기간을 연장하지 않기로 결정한 것에 실망했지만, 이후 이루어질 수 있다는 희망을 표명했습니다." |
    | KORani-v3-13B | "이스라엘 지난 주에는 미국 대통령 도널드 트럼프가 제한 없이 영사관을 이스라엘 정부로 이전하기 위해 신청서를 제출하지 않음으로써 이스라엘을 실망시켰지만, 이스라엘은 미국이 나중에 이전을 진행할 수 있기를 바라며 희망을 표명했습니다." |
    | GPT4 | "목요일 이스라엘은 미국 대통령 도널드 트럼프가 텔아비브에서 예루살렘으로 이스라엘 미국 대사관 이전을 지연시키기 위해 면제 서명을 하는 결정에 실망감을 표현했지만, 나중에 이동이 이루어질 수 있기를 희망한다고 말했습니다." |
    | DeepL | "이스라엘은 목요일 도널드 트럼프 미국 대통령이 이스라엘 주재 미국 대사관을 텔아비브에서 예루살렘으로 이전하는 것을 연기하는 면제에 서명하기로 한 결정에 대해 유감을 표명했지만 나중에 이전이 이루어질 수 있기를 희망한다고 밝혔다." |
    | ChatGPT 3.5 | "이스라엘은 목요일 미국 대통령 도널드 트럼프가 이스라엘의 대사관을 텔아비브에서 예루살렘으로 이전하는 것을 연기하기 위해 면제서에 서명한 결정에 실망을 표명했지만, 나중에 이사가 가능할 것을 희망한다고 밝혔습니다." |
    | Vicuna 13B | "이스라엘은 주요 보고서에 기꺼이 분노하여 트럼프 대통령이 지정한 국가의 대통령이라는 자신을 인정하지 못하고 그룹의 선언문을 체포하고 싶다고 밝혔습니다." |
    | Koalpaca-13B | "이스라엘은 미국 대통령 도널드 트럼프이 예루살렘을 이스라엘의 수도로 인정한 결정에 대해 실망을 표명하며, 추후에 대사관 이전을 희망한다고 표명했다." |


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

## Evaluation
We tested model performance using GPT-4, and the code and results of the test can be found through the [AutoEvalGPT](https://github.com/krafton-ai/AutoEvalGPT).

## Limitations
The Korean performance of our models is not as good as the English performance of [Vicuna](https://github.com/lm-sys/FastChat). We believe this is due to the not enough quality of foundation models in the Korean tasks (compared to Llama in English tasks) and the dataset quality, which is primarily translational. We will continue to update the new versions of the Korani models as soon as we achieve better results.

## License
Our github repo and models are intended for research purpose, non-commercial use only, subject to the model License of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us If you find any potential violation.
The code is released under the Apache License 2.0.
