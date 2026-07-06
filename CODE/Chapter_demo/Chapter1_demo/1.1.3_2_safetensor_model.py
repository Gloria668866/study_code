from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\models\Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

prompt ="中国的首都是？"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs=inputs["input_ids"],attention_mask=inputs["attention_mask"], max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, no_repeat_ngram_size=2)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)