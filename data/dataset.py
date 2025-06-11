from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages = []

        for entry in dataset:
            messages = entry.get('messages', [])
            text = self._extract_text(messages)
            if text.strip():
                self.messages.append(text)

    def _extract_text(self, messages):
        if isinstance(messages, list):
            return " ".join([msg['content'] for msg in messages if 'content' in msg])
        return ""

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = self.messages[idx]
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return encoded.input_ids.squeeze(), encoded.attention_mask.squeeze()