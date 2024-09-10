
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import time


st.title("Qwen2-0.5B Instruct Lora Text Generation")
st.write("Generate text using the Qwen2-0.5B-Instruct model fine-tuned with LoRA. Enter a prompt and get streaming text output.")


@st.cache_resource
def load_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "SHASWATSINGH3101/Qwen2-0.5B-Instruct_lora_merge",
            device_map='auto',      
            torch_dtype=torch.float16,   
            low_cpu_mem_usage=True      
        )
        tokenizer = AutoTokenizer.from_pretrained("SHASWATSINGH3101/Qwen2-0.5B-Instruct_lora_merge")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

model, tokenizer = load_model()

class FastTextStreamer:
    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.output_buffer = []
        self.generated_text = ""

    def put(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
 
        if isinstance(token_ids, list):
            if len(token_ids) > 0 and isinstance(token_ids[0], list):
                token_ids = [item for sublist in token_ids for item in sublist]
        else:
            token_ids = [token_ids]
        
        self.output_buffer.extend(token_ids)
        

        if len(self.output_buffer) >= 20:
            text = self.tokenizer.decode(self.output_buffer, skip_special_tokens=self.skip_special_tokens)
            self.generated_text += text
            st.write(text) 
            self.output_buffer = []

    def end(self):
        if self.output_buffer:
            text = self.tokenizer.decode(self.output_buffer, skip_special_tokens=self.skip_special_tokens)
            self.generated_text += text
            st.write(text)
        st.write("")  # New line at the end

def gen_streaming_optimized(model, prompt, maxlen=100, sample=True):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    streamer = FastTextStreamer(tokenizer)
    
  
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=maxlen,
            do_sample=sample,
            num_return_sequences=1,
            temperature=0.1,
            num_beams=1,
            top_p=0.95,
            streamer=streamer,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )


seed = 42
set_seed(seed)


if model is not None and tokenizer is not None:
    prompt = st.text_area("Enter your prompt:", value="")  # Empty prompt by default
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=259)

   
    if st.button("Generate Text"):
        if prompt.strip(): 
            with st.spinner("Generating text..."):
                in_data = f"Instruct: {prompt}\n{prompt}\nOutput:\n"
                start_time = time.time()
                gen_streaming_optimized(model, in_data, maxlen=max_tokens)
                end_time = time.time()
                st.success(f"Generation completed in {end_time - start_time:.2f} seconds.")
        else:
            st.warning("Please enter a prompt to generate text.")
else:
    st.error("The model could not be loaded. Please check the model path or Hugging Face availability.")
