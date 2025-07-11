import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
llama_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
    llama_model.device
)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = llama_model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))
print(llama_model)

h1 = llama_model.model.embed_tokens(input_ids)
h2 = llama_model.model.layers[0].self_attn.q_proj(h1)
h2, h2.shape