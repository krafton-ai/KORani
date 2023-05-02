import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, GenerationConfig
import torch
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import os, argparse

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.encounters = encounters
      self.counter = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
                # stop = torch.LongTensor(stop).to(device='cuda')
                if torch.all((stop == input_ids[0][-len(stop):])).item():
                    self.counter += 1
        if self.counter >= self.encounters:
            return True
        else:
            return False

####################################################################################
# python inference.py --model_path "KRAFTON/KORani-v3-13B"


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="translation")    
    parser.add_argument("--model_max_length", type=int, default=2048)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    ################################################################################
    with init_empty_weights():
        model = transformers.LlamaForCausalLM.from_pretrained(args.model_path)

    load_checkpoint_and_dispatch(
        model, 
        args.model_path, 
        device_map="auto",
        dtype='float16', 
    )

    ################################################################################
    model.eval()

    with open(f"prompts/{args.task}.txt") as f:
        prompt = "".join(f.readlines())

    batch = tokenizer(prompt, return_tensors="pt")
    prompt_size = len(batch['input_ids'][0])
    batch = {
        "input_ids" : batch['input_ids'],
        "attention_mask" : batch['attention_mask']
    }
    batch["input_ids"] = batch["input_ids"].cuda()


    stop_tokens = [[13, 2277, 29937, 12968, 29901]]
    stop_words_ids = [torch.tensor(stop_word).to(device='cuda', dtype=torch.int64) for stop_word in stop_tokens]
    encounters = 1
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters)])

    generation_config = GenerationConfig(
        temperature = 0.1,
        max_new_tokens = 512,
        exponential_decay_length_penalty = (1024, 1.03),
        eos_token_id = tokenizer.eos_token_id,
        repetition_penalty = 1.1,
        do_sample = True,
        top_p = 0.7,
        min_length = 5,
        use_cache = True,
        return_dict_in_generate = True,
    )

    generated = model.generate(batch["input_ids"], generation_config=generation_config, stopping_criteria=stopping_criteria)
    response = tokenizer.decode(generated['sequences'][0][prompt_size:], skip_special_tokens=True)
    print(response)