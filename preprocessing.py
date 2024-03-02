import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from typing import List


class PiiDataset(Dataset):
    def __init__(self,
                 dataframe,
                 tokenizer: BertTokenizerFast,
                 label_map: dict,
                 max_token_len: int = 512,
                 isTrain: bool = True
                 ):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_token_len = max_token_len
        self.isTrain = isTrain

    def encode_tokens(self, original_tokens: List[str], labels: List[str] = None):
        # Tokenize the text with BERT tokenizer and keep track of alignment
        tokenized_input = self.tokenizer(original_tokens,
                                         is_split_into_words=True,
                                         truncation=True,
                                         max_length=self.max_token_len,
                                         padding='max_length',
                                         return_tensors="pt")
        # Map tokens back to original words
        word_ids = tokenized_input.word_ids(batch_index=0)
        aligned_labels = []
        if labels is not None:
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:  # Special tokens or same word piece
                    # PyTorch's cross-entropy ignores indices with -100
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(self.label_map[labels[word_idx]])
                previous_word_idx = word_idx

        return tokenized_input['input_ids'], tokenized_input['attention_mask'], aligned_labels

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        tokens = row['tokens']
        if self.isTrain:
            labels = row['labels']
            input_ids, attention_mask, aligned_labels = self.encode_tokens(
                tokens, labels)
            return {
                "input_ids": input_ids.squeeze(),  # Remove batch dimension
                "attention_mask": attention_mask.squeeze(),  # Remove batch dimension
                "labels": torch.tensor(aligned_labels, dtype=torch.long)
            }
        else:
            input_ids, attention_mask, aligned_labels = self.encode_tokens(
                tokens, labels=None)
            return {
                "input_ids": input_ids.squeeze(),  # Remove batch dimension
                "attention_mask": attention_mask.squeeze(),  # Remove batch dimension
            }

    def __len__(self):
        return len(self.dataframe)


def pipeline(dataframe, label_map, max_token_len, isTrain, batch_size, shuffle):
    dataset = PiiDataset(dataframe,
                         tokenizer=BertTokenizerFast.from_pretrained(
                             "dslim/bert-base-NER"),
                         label_map=label_map,
                         max_token_len=max_token_len,
                         isTrain=isTrain)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader
