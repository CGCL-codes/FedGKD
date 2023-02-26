from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset


def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                       return_attention_mask=True, return_token_type_ids=True, return_tensors='pt', )
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])


class AG_news(Dataset):
    def __init__(self, split,max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', mirror='tuna',
                                                             do_lower_case=True, add_special_tokens=True,
                                                             pad_to_max_length=True,
                                                             truncation=True, padding='max_length',
                                                             max_length=max_length)
        dataset = load_dataset("./data/ag_news/", split=split)
        self.text = dataset['text']
        self.targets = dataset['label']
        self.max_length = max_length
        # tokenize(self.text,self.tokenizer)
        # self.tokenize(dataset['text'],self.tokenizer)

    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                            max_length=self.max_length, pad_to_max_length=True)
        return (inputs['input_ids'][0], inputs['attention_mask'][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.tokenize(self.text[index])
        target = self.targets[index]
        return input, target
