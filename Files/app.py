import os
import torch
import pandas as pd
import numpy as np
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set the Hugging Face home directory
os.environ['HF_HOME'] = '/app/.cache'

# Load the base model with device_map set to 'auto'
model = AutoModelForCausalLM.from_pretrained(
    "SHASWATSINGH3101/Qwen2-0.5B-Instruct_lora_merge",
    device_map='auto'
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("SHASWATSINGH3101/Qwen2-0.5B-Instruct_lora_merge")
tokenizer.pad_token = tokenizer.eos_token

def gen(model, p, maxlen=100, sample=True):
    toks = tokenizer(p, return_tensors="pt").to(model.device)
    res = model.generate(**toks, max_new_tokens=maxlen, do_sample=sample,
                         num_return_sequences=1, temperature=0.1, num_beams=1, top_p=0.95)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def generate_letter(prompt):
    seed = 42
    set_seed(seed)

    in_data = f"Instruct: {prompt}\n{prompt}\nOutput:\n"

    # Generate response
    peft_model_res = gen(model, in_data, 259)
    peft_model_output = peft_model_res[0].split('Output:\n')[1]

    # Extract the relevant parts of the output
    prefix, success, result = peft_model_output.partition('#End')

    return prefix.strip()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_letter,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs="text",
    title="Legal Letter Generator",
    description="Generate a letter informing someone of potential legal action due to a dispute or violation.",
    flagging_dir="/app/flagged"  # Set the flagging directory
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)
