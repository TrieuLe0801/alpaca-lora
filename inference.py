import os
import sys
from typing import Union, List
from convert_text_to_list import convert_text_to_list

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

default_instruction = """In this case, you are a genius doctor, based on your medical knowledge, let help me complete the exam correctly. What I want you do that read the Vietnamese question (multi-choice) and answer me in binary string where 1 is the choice you want me pick. Only give the answer, not contain any explaining.
Here are some examples that may help you understand how to solve my question:
Question: What are the symptoms of heart valve disease?
A. Difficulty breathing
B. Rapid weight gain
C. Jaundice
D. Hair loss
Answer:
1100

Question: Last August, An and Binh went for a health check-up. An was diagnosed with grade 3 myopia, Binh was diagnosed with fatty liver. How can Binh limit and reduce his illness?
A. Increase alcohol consumption
B. Eat a lot of foods containing cholesterol
C. Lose weight, exercise regularly and maintain a healthy diet
D. Smoking
Answer:
0010

Now, your turn:

Question:"""

prompt = """
In this case, you are a genius doctor, based on your medical knowledge. Please answer the letter of option truthfully.
"""

input_value = """
Question:
Mr. Than wants to check images of his heart and lungs. Which of the following methods can he use?
A. Chest X-ray
B. CT scan
C. Cardiac catheterization
D. Coronary angiography
E. Electrocardiogram (ECG)
Answer:
"""


def main(
    input_text: Union[str, List[str]] = input_value,
    instruction: str = default_instruction,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    # server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    # share_gradio: bool = False,
    
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    def evaluate(
        instruction,
        input=None,
        temperature=0.9,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        repetition_penalty=1.2,
        do_sample=True,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            # repetition_penalty=repetition_penalty, 
            do_sample=do_sample,
            # early_stopping=True,
            **kwargs,
        )

        # generate_params = {
        #     "input_ids": input_ids,
        #     "generation_config": generation_config,
        #     "return_dict_in_generate": True,
        #     "output_scores": True,
        #     "max_new_tokens": max_new_tokens,
        # }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return(output)
    if isinstance(input_text, str):
        output_generation = evaluate(instruction, input_text)
        print(output_generation)
    else:
        output_generation = []
        for i in input_text:
            output_text = evaluate(instruction, i)
            print(output_text)
            output_generation.append(output_text)
            
    return output_generation

if __name__ == "__main__":
    # fire.Fire(main)
    input_text_list = convert_text_to_list()
    output_list = main(
        input_text = input_text_list[5],
        load_8bit=True,
        base_model="medalpaca/medalpaca-7b",
        lora_weights="./lora-alpaca"
    )