from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pytreebank

class SST(Dataset):
    def __init__(self, split,max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', mirror='tuna',
                                                             do_lower_case=True, add_special_tokens=True,
                                                             pad_to_max_length=True,
                                                             truncation=True, padding='max_length',
                                                             max_length=max_length)
        dataset = pytreebank.load_sst("./data/sst/")
        if split == "valid":
            split = "dev"
        data = dataset[split]
        #data = dataset['ptb_tree']
        self.text = [tree.to_lines()[0] for tree in data]
        self.targets = [tree.label for tree in data]
        self.max_length = max_length

    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                            max_length=self.max_length, padding="max_length",truncation=True)
        return (inputs['input_ids'][0], inputs['attention_mask'][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.tokenize(self.text[index])
        target = self.targets[index]
        return input, target
