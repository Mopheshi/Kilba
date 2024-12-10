from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("lelapa/InkubaLM-0.4B", trust_remote_code=True)

text = "Today I planned to"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs.input_ids

# Create an attention mask
attention_mask = inputs.attention_mask

# Generate outputs using the attention mask
outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=60, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
