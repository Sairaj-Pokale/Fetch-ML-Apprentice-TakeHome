import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningDataModule
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sent_label_id = {"Positive": 0, "Negative": 1, "Neutral": 2}

sentiment_data = [
    "This movie was absolutely fantastic! The acting was superb and the storyline was gripping.",
    "I'm quite disappointed with the product quality; it broke after just one day of use.",
    "The weather today is partly cloudy with a chance of showers later in the afternoon.",
    "Customer service was incredibly helpful and resolved my issue in no time!",
    "The new software update seems a bit clunky and not very intuitive to navigate.",
]
sentiment_labels = ["Positive", "Negative", "Neutral", "Positive", "Negative"]
sent_labels = [sent_label_id[lbl] for lbl in sentiment_labels]
ner_data = [
    {
        "sentence": "Anna Wintour works for Vogue in New York City.",
        "tokens_tags": [
            ("Anna", "B-PER"),
            ("Wintour", "I-PER"),
            ("works", "O"),
            ("for", "O"),
            ("Vogue", "B-ORG"),
            ("in", "O"),
            ("New", "B-LOC"),
            ("York", "I-LOC"),
            ("City", "I-LOC"),
            (".", "O"),
        ],
    },
    {
        "sentence": "The G7 summit will be held in London next June.",
        "tokens_tags": [
            ("The", "O"),
            ("G7", "B-ORG"),
            ("summit", "O"),
            ("will", "O"),
            ("be", "O"),
            ("held", "O"),
            ("in", "O"),
            ("London", "B-LOC"),
            ("next", "B-MISC"),
            ("June", "I-MISC"),
            (".", "O"),
        ],
    },
    {
        "sentence": "Elon Musk announced that SpaceX will launch Starship next month.",
        "tokens_tags": [
            ("Elon", "B-PER"),
            ("Musk", "I-PER"),
            ("announced", "O"),
            ("that", "O"),
            ("SpaceX", "B-ORG"),
            ("will", "O"),
            ("launch", "O"),
            ("Starship", "B-MISC"),
            ("next", "O"),
            ("month", "O"),
            (".", "O"),
        ],
    },
    {
        "sentence": "The FIFA World Cup is a global football tournament organized by FIFA.",
        "tokens_tags": [
            ("The", "O"),
            ("FIFA", "B-ORG"),
            ("World", "I-MISC"),
            ("Cup", "I-MISC"),
            ("is", "O"),
            ("a", "O"),
            ("global", "O"),
            ("football", "O"),
            ("tournament", "O"),
            ("organized", "O"),
            ("by", "O"),
            ("FIFA", "B-ORG"),
            (".", "O"),
        ],
    },
    {
        "sentence": "Barack Obama visited Berlin during his presidency.",
        "tokens_tags": [
            ("Barack", "B-PER"),
            ("Obama", "I-PER"),
            ("visited", "O"),
            ("Berlin", "B-LOC"),
            ("during", "O"),
            ("his", "O"),
            ("presidency", "O"),
            (".", "O"),
        ],
    },
]


class SentimentDatasetClass(Dataset):
    def __init__(self, sentiment_data, sentiment_labels):
        self.texts = sentiment_data
        self.labels = sentiment_labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.max_length = 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["task"] = "SEN"
        return item


class NERDatasetClass(Dataset):
    def __init__(self, ner_data, ner_ignore_idx):
        self.ner_label_to_id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-LOC": 3,
            "I-LOC": 4,
            "B-ORG": 5,
            "I-ORG": 6,
            "B-MISC": 7,
            "I-MISC": 8,
        }
        self.id_to_ner_label = {v: k for k, v in self.ner_label_to_id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.ner_ignore_idx = ner_ignore_idx
        self.data = []
        for item in ner_data:
            tokens, tags = zip(*item["tokens_tags"])
            tag_ids = [self.ner_label_to_id[tag] for tag in tags]
            self.data.append({"tokens": list(tokens), "labels": tag_ids, "task": "NER"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        labels = item["labels"]
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        word_ids = encoded.word_ids(0)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(self.ner_ignore_idx)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
        previous_word_idx = word_idx
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(aligned_labels)
        item["task"] = "NER"
        return item


class MultiTaskSentenceTransformer(pl.LightningModule):
    """
    MultiTask Learning
    Task A: Sentiment Analysis -> Classify sentences/reviews into predefined sentiment classes such as Happy, Sad, & Neutral
    Task B: Named Entity Recognition -> Classify words of the sentences into prefined entity classes such as Person, Organization, Location etc.
    TODO: Describe the changes made to the architecture to support multi-task learning
    -- Added individual task specific heads to the model
    -- Updated forward to guide the logits obtained from the backbone to the respective task heads
    -- Tried out PyTorch Lightning, a pytorch wrapper, so added the training_step to declare a single training step, same for validation and test
    -- To accomodate the flow of the framework, the loss function and optimizer is declared in the architecture it self
    -- Added task specific metrics, Accuracy for sentiment analysis and F1 Score for named entity recognition
    """

    def __init__(
        self,
        sentiment_classes,
        ner_classes,
        dropout_rate,
        embedding_dim,
        ner_ignore_idx,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.sentiment_analysis_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, sentiment_classes),
        )
        self.named_entity_rec_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, ner_classes),
        )
        self.sent_crit = nn.CrossEntropyLoss()
        self.ner_crit = nn.CrossEntropyLoss(ignore_index=ner_ignore_idx)
        self.ner_classes = ner_classes
        self.sentiment_acc = Accuracy(task="multiclass", num_classes=sentiment_classes)
        self.ner_f1 = F1Score(
            task="multiclass",
            num_classes=ner_classes,
            average="macro",
            ignore_index=ner_ignore_idx,
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, x, task, pooler_type=None):
        model_output = self.backbone(**x)
        if task == "SEN":
            if pooler_type == "native":
                pooled_output = model_output.pooler_output
            elif pooler_type == "mean":
                pooled_output = self.mean_pooling(model_output, x["attention_mask"])
                pooled_output = F.normalize(pooled_output, p=2, dim=1)
            outputs = self.sentiment_analysis_head(pooled_output)
        if task == "NER":
            outputs = self.named_entity_rec_head(model_output.last_hidden_state)
        return outputs

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable_params, lr=3e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        item = batch
        task = item.pop("task")[0]
        labels = item.pop("labels")
        pooler_type = "mean"
        outputs = self(item, task, pooler_type)
        print(task)
        if task == "SEN":
            loss = self.sent_crit(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = self.sentiment_acc(preds, labels)
            self.log("train/sentiment_acc", acc, on_step=False, on_epoch=True)
        if task == "NER":
            loss = self.ner_crit(outputs.view(-1, self.ner_classes), labels.view(-1))
            preds = torch.argmax(outputs, dim=-1)
            f1 = self.ner_f1(preds.view(-1), labels.view(-1))
            self.log("train/ner_f1", f1, on_step=False, on_epoch=True)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pooler_type = "mean"
        item = batch
        task = item.pop("task")[0]
        labels = item.pop("labels")
        outputs = self(item, task, pooler_type)
        if task == "SEN":
            loss = self.sent_crit(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = self.sentiment_acc(preds, labels)
            self.log("val/sentiment_acc", acc, on_step=False, on_epoch=True)
        if task == "NER":
            loss = self.ner_crit(outputs.view(-1, self.ner_classes), labels.view(-1))
            preds = torch.argmax(outputs, dim=-1)
            f1 = self.ner_f1(preds.view(-1), labels.view(-1))
            self.log("val/ner_f1", f1, on_step=False, on_epoch=True)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        pooler_type = "mean"
        item = batch
        task = item.pop("task")[0]
        labels = item.pop("labels")
        outputs = self(item, task, pooler_type)
        if task == "SEN":
            loss = self.sent_crit(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = self.sentiment_acc(preds, labels)
            self.log("test/sentiment_acc", acc, on_step=False, on_epoch=True)
        if task == "NER":
            loss = self.ner_crit(outputs.view(-1, self.ner_classes), labels.view(-1))
            preds = torch.argmax(outputs, dim=-1)
            f1 = self.ner_f1(preds.view(-1), labels.view(-1))
            self.log("test/ner_f1", f1, on_step=False, on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss


class NERDataModule(LightningDataModule):
    def __init__(self, ner_data, ner_ignore_idx, batch_size):
        super().__init__()
        self.ner_data = ner_data
        self.ner_ignore_idx = ner_ignore_idx
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = NERDatasetClass(self.ner_data, self.ner_ignore_idx)
        self.val_dataset = NERDatasetClass(self.ner_data, self.ner_ignore_idx)
        self.test_dataset = NERDatasetClass(self.ner_data, self.ner_ignore_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class SentimentDataModule(LightningDataModule):
    def __init__(self, sentiment_data, sentiment_labels, batch_size):
        super().__init__()
        self.train_texts = sentiment_data
        self.train_labels = sentiment_labels
        self.val_texts = sentiment_data
        self.val_labels = sentiment_labels
        self.test_texts = sentiment_data
        self.test_labels = sentiment_labels
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SentimentDatasetClass(self.train_texts, self.train_labels)
        self.val_dataset = SentimentDatasetClass(self.val_texts, self.val_labels)
        self.test_dataset = SentimentDatasetClass(self.test_texts, self.test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


multi_model = MultiTaskSentenceTransformer(
    sentiment_classes=3,
    ner_classes=9,
    dropout_rate=0.1,
    embedding_dim=384,
    ner_ignore_idx=-100,
)

trainer_config = {
    "max_epochs": 20,
    "precision": (
        "16-mixed"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else 32
    ),
    "logger": pl.loggers.TensorBoardLogger("training_logs/", name="mtl_sentiment_ner"),
    "callbacks": [
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        pl.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=3),
    ],
}

if torch.cuda.is_available():
    trainer_config["accelerator"] = "gpu"
    trainer_config["devices"] = 1
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    trainer_config["accelerator"] = "cpu"
    print("Training on CPU")

trainer = pl.Trainer(**trainer_config)

sentiment_dm = SentimentDataModule(
    sentiment_data=sentiment_data, sentiment_labels=sent_labels, batch_size=2
)

trainer.fit(multi_model, datamodule=sentiment_dm)
trainer.test(multi_model, datamodule=sentiment_dm)

ner_dm = NERDataModule(ner_data=ner_data, ner_ignore_idx=-100, batch_size=2)

trainer.fit(multi_model, datamodule=ner_dm)
trainer.test(multi_model, datamodule=ner_dm)
