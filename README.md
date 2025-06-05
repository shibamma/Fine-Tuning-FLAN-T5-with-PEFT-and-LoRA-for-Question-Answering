# Fine-Tuning-FLAN-T5-with-PEFT-and-LoRA-for-Question-Answering
Fine-tuned google/flan-t5-base using Low-Rank Adaptation (LoRA) for a custom question-answering task, demonstrating effective use of parameter-efficient techniques on large language models.
Implemented PEFT (Parameter-Efficient Fine-Tuning) pipeline using Hugging Face Transformers and PEFT library, reducing trainable parameters by over 98\% while maintaining performance.
Evaluated model performance using ROUGE metrics, showing measurable improvements over the base model in ROUGE-1 (+0.05), ROUGE-2 (+0.04), and ROUGE-L (+0.06).
Designed and pre-processed a custom context-based QA dataset, tokenized using a prompt-based approach suitable for sequence-to-sequence learning.
Conducted qualitative and quantitative analysis of model outputs, validating LoRA’s effectiveness in resource-constrained environments.

Model link: https://huggingface.co/google/flan-t5-base/tree/main
