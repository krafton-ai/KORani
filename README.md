# KORani

- KORani: Large Language Models for 🇰🇷 Korean and 🇺🇸 English using LLaMA 13B and Polyglot 12.8B.
- Tested which LLM is effective for 🇰🇷 Korean tasks after finetuning.
- 🤗 You can download the models from the [Link](https://huggingface.co/KRAFTON).

### Release
<p align="center">
<a href=""><img src="assets/KORani.png" width="20%"></a>
</p>
 
This repository contains inference code for KORani models that are based on [LLaMA 13B](https://arxiv.org/abs/2302.13971v1) and [Polyglot 12.8B](https://huggingface.co/EleutherAI/polyglot-ko-12.8b).
KORani models are finetuned using [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) & [KoVicuna](https://github.com/melodysdreamj/KoVicuna).
We offer three types of models as follows.
- 1️⃣ KORani-v1-13B: finetuned using KoVicuna dataset based on Polyglot 12.8B <https://huggingface.co/KRAFTON/KORani-v1-13B>
- 2️⃣ KORani-v2-13B: finetuned using KoVicuna dataset based on LLaMA 13B <https://huggingface.co/KRAFTON/KORani-v2-13B>
- 3️⃣ KORani-v3-13B: finetuned using ShareGPT & KoVicuna datset based on LLaMA 13B <https://huggingface.co/KRAFTON/KORani-v3-13B>

* We used LLaMA 13B from [here](https://huggingface.co/decapoda-research/llama-13b-hf).
* We extracted only the data from [Kovicuna](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) that corresponds to the first and second parts of the conversation, which are 'human' and 'GPT'.
* The model finetuning was conducted on eight A100 40GB GPUs. The code used for training is based on the [Link](https://github.com/lm-sys/FastChat).

### Local Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Inference
0. Command <br>
    `--model_path` (str): model path for evaluation. (e.g. KRAFTON/KORani-v3-13B) <br>
    `--task` (str): choose which task you want to evaluate. (e.g. QA) <br>


1. Question Answering

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


2. Translation

    ```bash
    python inference.py \
        --model_path "KRAFTON/KORani-v3-13B"
        --task "translation"
    ```

    Below is the prompt used to generate the answer. You can modify it in the [translation link](prompts/translation.txt). If you use few-shot in translation, the performance can improve.

    ```python
    PROMPT = """\
    ### Instruction: Translate Korean sentence into English. You may leave specific names as they are.
    Korean: 얼마나 많은 언어를 말할 수 있니?
    English: How many languages can you speak?#
    Korean: 일 다 끝났어?
    English: Did you finish your work?#
    Korean: {Target_sentence} 
    English:"""
    ```


3. Summarization

    ```bash
    python inference.py \
        --model_path "KRAFTON/KORani-v3-13B"
        --task "summarization"
    ```

    Below is the prompt used to generate the answer. You can modify it in the [summarization link](prompts/summarization.txt). It does not work for a max length of over 2048.

    ```python
    PROMPT = """\
    # Meeting note
    {Target_document}

    # Summarize the meeting note into 3 Korean sentences.
    ### Output: 1)"""
    ```

### Evaluation
We tested model performance using GPT-4, and the code and results of the test can be found through the [AutoEvalGPT](https://github.com/krafton-ai/AutoEvalGPT).

### License
Our github repo and models are intended for research purpose, non-commercial use only, subject to the model License of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us If you find any potential violation.
The code is released under the Apache License 2.0.

### Contact
Gibbeum Lee (pirensisco@krafton.com) <br>
Seongjun Yang (seongjunyang@krafton.com)