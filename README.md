# ECE-GY-7123-NewsClassification_LoRA_Finetuning
Overview
This project implements a parameter-efficient approach to fine-tune RoBERTa-base for news text classification using Low-Rank Adaptation (LoRA). Using the AG News dataset, we demonstrate that by adapting less than 1% of the model's parameters, we can achieve comparable or better performance than full fine-tuning (95.3% accuracy) with significantly reduced computational requirements.
Key Features

Parameter-Efficient Fine-Tuning: We use LoRA to train only 0.6% of the model parameters (~741K vs. 125M)
Contextual Data Augmentation: Enhanced training with MLM-based augmentation for improved generalization
Mixed-Precision Training: FP16 acceleration for faster training and reduced memory footprint
High Performance: 95.3% accuracy on AG News classification, matching or exceeding full fine-tuning

Repository Contents

NewsClassification_LoRA_Finetuning.ipynb: Main Jupyter notebook with all code implementation
inference_output.csv: Model predictions on the test dataset
test_unlabelled.pkl: Unlabelled test dataset for inference

Technical Details
LoRA Configuration

Rank (r): 4
Alpha (α): 16
Target modules: Query and Value projection matrices
LoRA dropout: 0.05
Fine-tuned classification head

Training Setup

Learning rate: 4e-4
Batch size: 32 (with gradient accumulation for effective batch size of 64)
Optimizer: AdamW with weight decay 0.01
Scheduler: Cosine decay with 10% warmup
Early stopping with patience of 3 epochs
Mixed precision (FP16)

Results
Our model achieves:

95.3% accuracy on the AG News test set
95.3% macro F1 score
Per-class F1: World (96.1), Sports (94.6), Business (94.8), Sci/Tech (95.6)

The LoRA approach delivered equivalent or better performance compared to full fine-tuning while being approximately 2× faster and using significantly less memory.
Requirements
transformers
datasets
peft
torch
evaluate
scikit-learn
matplotlib
pandas
Usage

Open NewsClassification_LoRA_Finetuning.ipynb in Google Colab or Jupyter
Run all cells to train the model and generate predictions
The notebook is fully documented with explanations at each step

Contributors

Shubham Naik (svn9724@nyu.edu)
Shruti Pangare (stp8232@nyu.edu)
Rudra Patil (rp4216@nyu.edu)

References

Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
Wu, X., et al. (2018). Conditional BERT Contextual Augmentation.
Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification.

