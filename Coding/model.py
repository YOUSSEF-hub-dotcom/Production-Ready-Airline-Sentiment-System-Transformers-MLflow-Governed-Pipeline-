# model.py

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nltk
import pandas as pd
import logging

logger = logging.getLogger("Model")


# --- 1. Custom Dataset for Lazy Loading ---
class AirlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    # Implementation inside the Dataset class
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_and_train_distilbert_model(df, epochs, lr, batch_size):
    logger.info("===========================>>>> Deep Learning (DistilBERT) Professional Pipeline")

    nltk.download(['averaged_perceptron_tagger', 'punkt', 'stopwords', 'wordnet', 'omw-1.4'], quiet=True)
    df['label'] = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    df = df[['cleaned_text', 'label']].copy()

    # --- 2. Split (Train, Val, Test) ---
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Augmentation (For training data only)
    aug = naw.SynonymAug(aug_src='wordnet')
    max_count = train_df['label'].value_counts().max()
    augmented_texts, augmented_labels = [], []
    for label in train_df['label'].unique():
        subset = train_df[train_df['label'] == label]
        augmented_texts.extend(subset['cleaned_text'].tolist())
        augmented_labels.extend(subset['label'].tolist())
        needed = max_count - len(subset)
        if needed > 0:
            gen = set()
            while len(gen) < needed:
                txt = aug.augment(subset.sample(1).iloc[0]['cleaned_text'])
                if isinstance(txt, list): txt = " ".join(txt)
                gen.add(txt)
            augmented_texts.extend(list(gen))
            augmented_labels.extend([label] * len(gen))

    balanced_train_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})

    MAX_LEN = 512
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Prepare DataLoaders using the Custom Dataset
    train_ds = AirlineDataset(balanced_train_df['text'].values, balanced_train_df['label'].values, tokenizer, MAX_LEN)
    val_ds = AirlineDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)
    test_ds = AirlineDataset(test_df['cleaned_text'].values, test_df['label'].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

    # --- 3. Optimizer & Scheduler (Warmup) ---
    # Class weights slightly adjusted because of Augmentation
    weights = torch.tensor([1.0, 2.0, 1.5]).to(device)  
    loss_fn = CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * total_steps)  # 10% Warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    progress_bar = tqdm(range(total_steps), desc="Training")

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping for weight stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Update Learning Rate

            epoch_loss += loss.item()
            progress_bar.update(1)

        # Validation Step after each Epoch
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_acc += (preds == labels).sum().item()

        logger.info(
            f"Epoch {epoch + 1} | Val Acc: {val_acc / len(val_df):.4f} | Avg Loss: {epoch_loss / len(train_loader):.4f}")

    progress_bar.close()

    # --- Final Evaluation on Test Set ---
    logger.info("\nEvaluating on final test set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive'])

    logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + report)

    return {
        'model': model, 'tokenizer': tokenizer, 'test_accuracy': accuracy,
        'test_predictions': all_preds, 'test_labels': all_labels,
        'device': device, 'params': {'epochs': epochs, 'lr': lr, 'batch_size': batch_size}
    }
