import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

def test():
	test_data = pd.read_csv("tuning-meta-llms-for-african-language-machine-translation/Test.csv")
	model_name = "KevinKibe/nllb-200-distilled-1.3B-finetuned-finetuned"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	# Define the function to translate texts
	def translate_texts(texts, batch_size=64):
		def process_batch(batch):
			inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
			outputs = model.generate(inputs["input_ids"])
			translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
			return translations

		num_batches = int(np.ceil(len(texts) / batch_size))
		all_translations = []

		for i in range(num_batches):
			batch = texts[i * batch_size:(i + 1) * batch_size]
			translations = process_batch(batch)
			all_translations.extend(translations)

		return all_translations

	# Get translations
	translations = translate_texts(test_data["English"].tolist())

	# Add the translations to a new column
	test_data["Twi"] = translations

	test_data[["id", "Twi"]].to_csv("submission_1.csv", index=False)

test()