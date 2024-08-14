import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

df = pd.read_csv('related_topics.csv')

texts = df['Data'].tolist()
labels = df['Label'].apply(lambda x: 0 if x == 'Not Related' else 1).tolist()
tr_texts, val_texts, tr_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

tr_enc = tokenizer(tr_texts, truncation=True, padding=True, max_length=128)
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128)


class TxtDataset(Dataset):
    def __init__(self, enc, lbls):
        self.enc = enc
        self.lbls = lbls

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.enc.items()}
        item['labels'] = torch.tensor(self.lbls[idx])
        return item

    def __len__(self):
        return len(self.lbls)

tr_ds = TxtDataset(tr_enc, tr_labels)

val_ds = TxtDataset(val_enc, val_labels)

args = TrainingArguments(
    output_dir='./res',
    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,

    logging_dir='./logs',
    logging_steps=10,

    evaluation_strategy="steps",
    save_steps=10,

    eval_steps=10,

    load_best_model_at_end=True,
)

class MyTrainer(Trainer):
    def compute_metrics(self, pred):
        lbls = pred.label_ids

        preds = pred.predictions.argmax(-1)
        
        acc = (preds == lbls).mean()
        return {"accuracy": acc}

trainer = MyTrainer(
    model=model,
    args=args,
    train_dataset=tr_ds,
    eval_dataset=val_ds,
)

trainer.train()

eval_res = trainer.evaluate(eval_dataset=val_ds)
print(f"Eval results: {eval_res}")

model.save_pretrained('./ft_model')
tokenizer.save_pretrained('./ft_tokenizer')
