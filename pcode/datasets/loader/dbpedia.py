from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import numpy as np

def tokenize(
    examples,
    tokenizer,
    max_seq_len,
    pad_token=0,
    pad_token_segment_id=0,
):
    """
    task: the name of one of the glue tasks, e.g., mrpc.
    examples: raw examples, e.g., common.SentenceExamples.
    tokenizer: BERT/ROBERTA tokenizer.
    max_seq_len: maximum sequence length of the __word pieces__.
    label_list: list of __the type__ of gold labels, e.g., [0, 1].

    mzhao: I made following __default__ options to avoid useless stuff:
        (i) pad the sequence from right.
        (ii) attention masking:
            1 -> real tokens
            0 -> [PAD]
        (iii) i skip the only one regression task in glue sts-b.
    """
    assert pad_token == pad_token_segment_id == 0

    # associate each label with an index

    features = []
    print("[INFO] *** Convert Example to Features ***")
    for idx, eg in enumerate(tqdm(examples)):
        # inputs:
        # input_ids: list[int],
        # token_type_ids: list[int] if return_token_type_ids is True (default)
        # attention_mask: list[int] if return_attention_mask is True (default)
        # overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
        # num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
        # special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
        # NOTE: [SEP] belongs to text_a


        inputs = tokenizer.encode_plus(
            eg, add_special_tokens=True, max_length=max_seq_len
        )

        # these stuff are not padded
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)

        # pad everything to max_seq_len
        padding_len = max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_token] * padding_len
        attention_mask = attention_mask + [0] * padding_len


        assert (
            len(input_ids) == len(attention_mask) == max_seq_len
        ), "{} - {} - {}".format(len(input_ids), len(attention_mask))

        features.append((input_ids,attention_mask))

    return features


class dbpedia_14(Dataset):
    def __init__(self, split,max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', mirror='tuna',
                                                             do_lower_case=True, add_special_tokens=True,
                                                             pad_to_max_length=True,return_attention_mask=True,
                                                            return_token_type_ids=True,
                                                             truncation=True, padding='max_length',
                                                             max_length=max_length)
        dataset = load_dataset("/home/dzyao/wnpan/feddf/data/dbpedia_14", split=split)
        self.text = dataset['content']
        self.targets = dataset['label']
        self.max_length = max_length
        self.features = []
        #self.features = tokenize(self.text,self.tokenizer,self.max_length)
        #self.tokenize_all(self.text,self.tokenizer)

        # self.tokenize(dataset['text'],self.tokenizer)


    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                            max_length=self.max_length, pad_to_max_length=True)
        return (inputs['input_ids'][0], inputs['attention_mask'][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if len(self.features) > 0:
            input = self.features[index]
        else:
            input = self.tokenize(self.text[index])
        target = self.targets[index]
        return input, target

dataset = dbpedia_14('train')