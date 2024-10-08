from utils import load_dataset, load_model, split
from transformers import DataCollatorForSeq2Seq
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()


model_name = os.getenv("MODEL_NAME")
wandb_api_key = os.getenv("WANDB_API_KEY")
read_token = os.getenv("HUGGINGFACE_READ_API_KEY")
write_token = os.getenv("HUGGINGFACE_WRITE_API_KEY")

def main():
	train_data, test_data = load_dataset(train_path = "tuning-meta-llms-for-african-language-machine-translation/Train.csv", test_path=  "tuning-meta-llms-for-african-language-machine-translation/Test.csv")
	source = train_data["English"].values.tolist()
	target = train_data["Twi"].values.tolist()
	tokenizer, model, processor = load_model(model_name, read_token)
	train_dataset, test_dataset = split(tokenizer, source, target)
	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
	metric = evaluate.load("rouge")

	def postprocess_text(preds, labels):
		preds = [pred.strip() for pred in preds]
		labels = [[label.strip()] for label in labels]
		return preds, labels

	def compute_metrics(eval_preds):
		preds, labels = eval_preds
		if isinstance(preds, tuple):
			preds = preds[0]
		decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
		decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
		result = metric.compute(predictions=decoded_preds, references=decoded_labels)
		rouge_score = result["rougeL"]
		prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
		result = {
			"rouge": round(rouge_score, 4),
			"gen_len": np.mean(prediction_lens)
		}
		return result
	training_args = Seq2SeqTrainingArguments(
		output_dir=f"../{model_name}-finetuned",
		optim="adamw_bnb_8bit",
		evaluation_strategy="steps",
		save_strategy='steps',
		max_steps=10,
		save_steps=5,
		eval_steps=5,
		logging_steps=5,
		learning_rate=5e-5,
		lr_scheduler_type="cosine",
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		weight_decay=0.01,
		save_total_limit=3,
		num_train_epochs=20,
		predict_with_generate=True,
		load_best_model_at_end=True,
		fp16=True,
		push_to_hub=True,
		hub_token=write_token,
		hub_strategy="checkpoint",
		save_safetensors=False,
		resume_from_checkpoint="last-checkpoint",
		report_to="wandb",
	)

	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)
	tokenizer.save_pretrained(training_args.output_dir)
	processor.save_pretrained(training_args.output_dir)
	trainer.train()
	trainer.push_to_hub()

main()