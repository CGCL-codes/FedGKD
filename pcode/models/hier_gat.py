import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["hier_gat"]
class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(AttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)


    def forward(self, words, word_emb, attr_emb):
        words_emb = word_emb(words)
        attr_emb = attr_emb.unsqueeze(1)
        attrs_emb = attr_emb.repeat(1, words_emb.size()[1], 1)
        combina = torch.cat([words_emb, attrs_emb], dim=2)

        e = self.leakyrelu(torch.matmul(combina, self.a)).squeeze(-1)  # (batch size, seq length)
        attn = torch.zeros(words_emb.size()[0], word_emb.num_embeddings)  # (batch size, vocab length)
        attn = attn.to(words.device)

        for i in range(words_emb.size()[0]):
            attn[i][words[i]] = e[i]

        return attn


class ContAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(ContAttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, attrs, all):
        alls = all.repeat(attrs.size()[0], 1)
        combina = torch.cat([attrs, alls], dim=1)

        e = self.leakyrelu(torch.matmul(combina, self.a))
        attention = F.softmax(e, dim=0)

        return attrs - attention * alls


class GlobalAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(GlobalAttentionLayer, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, words_emb):
        words_emb = self.linear(words_emb)

        e = self.leakyrelu(torch.matmul(words_emb, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1).unsqueeze(1)

        attributes_emb = torch.matmul(attention, words_emb).squeeze(1)
        return F.relu(attributes_emb)


class StructAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(StructAttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, attrs_emb, entity_emb):
        attr_num = attrs_emb.size()[1]

        entity_emb = entity_emb.unsqueeze(1)
        entitys_emb = entity_emb.repeat(1, attr_num, 1)
        combina = torch.cat([attrs_emb, entitys_emb], dim=2)

        e = self.leakyrelu(torch.matmul(combina, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1).unsqueeze(1) * attr_num

        entitys_emb = torch.matmul(attention, attrs_emb).squeeze(1)
        return entitys_emb


class ResAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha, thr=0.5):
        super(ResAttentionLayer, self).__init__()

        self.thr = thr

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, entity_embs):
        Wh = self.linear(entity_embs)

        a_input = self._prepare_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1)

        # We apply the pooling operation
        attention = (attention < self.thr).type(attention.dtype) * attention
        h_prime = torch.matmul(attention, Wh)

        return F.elu(entity_embs - h_prime)

    def _prepare_input(self, Wh):
        N = Wh.size()[0]
        d = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * d)

def get_lm_path(lm, lm_path):
    if lm_path != None:
        return lm_path

    if lm == 'bert':
        return 'bert-base-uncased'
    elif lm == 'distilbert':
        return 'distilbert-base-uncased'
    elif lm == 'roberta':
        return 'roberta-base'
    elif lm == 'xlnet':
        return 'xlnet-base-cased'

class TranHGAT(nn.Module):
    def __init__(self, attr_num, finetuning=True, lm='bert', lm_path=None):
        super().__init__()

        # load the model or model checkpoint
        path = get_lm_path(lm, lm_path)
        self.lm = lm
        if lm == 'bert':
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetModel
            self.bert = XLNetModel.from_pretrained(path)

        self.finetuning = finetuning

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GlobalAttentionLayer(hidden_size, 0.2)
            for _ in range(attr_num)])
        self.conts = nn.ModuleList([
            AttentionLayer(hidden_size + hidden_size, 0.2)
            for _ in range(attr_num)])
        self.out = StructAttentionLayer(hidden_size * (attr_num + 1), 0.2)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, input):
        xs, masks = input
        xs = xs.permute(1, 0, 2) #[Attributes, Batch, Tokens]
        masks = masks.permute(0, 2, 1) # [Batch, All Tokens, Attributes]

        attr_outputs = []
        pooled_outputs = []
        attns = []
        if self.training and self.finetuning:
            self.bert.train()
            for x, init, cont in zip(xs, self.inits, self.conts):
                attr_embeddings = init(self.bert.get_input_embeddings()(x)) # [Batch, Hidden]
                attr_outputs.append(attr_embeddings)

                attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings) # [Batch, All Tokens]
                attns.append(attn)

            attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks # [Batch, All Tokens, Attributes]
            attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2) # [Batch, Attributes, Hidden]
            for x in xs:
                if self.lm == 'distilbert':
                    words_emb = self.bert.embeddings(x)
                else:
                    words_emb = self.bert.get_input_embeddings()(x)

                for i in range(words_emb.size()[0]): # i is index of batch
                    words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                output = self.bert(inputs_embeds=words_emb)
                pooled_output = output[0][:, 0, :]
                pooled_output = self.dropout(pooled_output)
                pooled_outputs.append(pooled_output)

            attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
            entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
            entity_output = self.out(attr_outputs, entity_outputs)
        else:
            self.bert.eval()
            with torch.no_grad():
                for x, init, cont in zip(xs, self.inits, self.conts):
                    attr_embeddings = init(self.bert.get_input_embeddings()(x))
                    attr_outputs.append(attr_embeddings)

                    # 64 * 768
                    attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)
                    attns.append(attn)

                attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks
                attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)
                for x in xs:
                    if self.lm == 'distilbert':
                        words_emb = self.bert.embeddings(x)
                    else:
                        words_emb = self.bert.get_input_embeddings()(x)

                    for i in range(words_emb.size()[0]):
                        words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                    output = self.bert(inputs_embeds=words_emb)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    pooled_outputs.append(pooled_output)

                attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
                entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
                entity_output = self.out(attr_outputs, entity_outputs)

        logits = self.fc(entity_output)
        return entity_output,logits

def hier_gat(conf):
    dataset = conf.data
    attr_num = 3

    if "Amazon-Google" in dataset:
        attr_num = 3
    model = TranHGAT(attr_num,lm = 'distilbert')

    return model