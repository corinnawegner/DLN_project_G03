import math
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from tokenizer import BertTokenizer
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf.
        # attention scores are calculated by multiplying queries and keys
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized) attention score between the j-th
        # and k-th token, given by i-th attention head before normalizing the scores,
        # use the attention mask to mask out the padding token scores.

        # Note again: in the attention_mask non-padding tokens are marked with 0 and
        # adding tokens with a large negative number.

        ### TODO
        bs, num_attention_heads, seq_len, attention_head_size = key.size()
        hidden_size = num_attention_heads * attention_head_size
        # compute the attention scores

        S = torch.matmul(query,key.transpose(-1,-2)) / math.sqrt(self.attention_head_size)
        # add attention mask to mask out padding tokens
        S = S + attention_mask
        # normalize the attention scores
        S = F.softmax(S,dim=-1)
        # apply dropout 
        S = self.dropout(S)
        # calculate the attention value
        V = torch.matmul(S,value)
        # recover the original shape
        #V = V.transpose(-2,-1).contiguous().view(bs,seq_len,hidden_size)

        V = V.transpose(1, 2).contiguous()
        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        V = V.view(*new_context_layer_shape)

        return V
        # raise NotImplementedError
        # Normalize the scores.
        # Multiply the attention scores to the value and get back V'.
        # Next, we need to concat multi-heads and recover the original shape
        # [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        Apply residual connection to any layer and normalize the output.
        This function is applied after the multi-head attention layer or the feed forward layer.

        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied
        ln_layer: the layer norm to be applied
        """
        ### TODO
         # use dense layer to transform the output
        output = dense_layer(output)
        # apply droupout
        output = dropout(output)
        # apply the residual connection by adding inout to output
        output = input + output
        # apply normalization
        output = ln_layer(output)
        return output
        # raise NotImplementedError
        # Hint: Remember that BERT applies dropout to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.

    def forward(self, hidden_states, attention_mask):
        """
        A single pass of the bert layer.

        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
        attention_mask: the mask for the attention layer

        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        ### TODO
        # pass progress
        mu_layer = self.self_attention(hidden_states,attention_mask)
        # add_norm 
        mu_layer = self.add_norm(hidden_states, mu_layer, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
        # pass progress
        ff_layer = self.interm_dense(mu_layer)
        # activate
        ff_layer = self.interm_af(ff_layer)
        # add_norm
        ff_layer = self.add_norm(mu_layer, ff_layer, self.out_dense,self.out_dropout,self.out_layer_norm)
        return ff_layer
        # raise NotImplementedError


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

        # Initialize the spaCy model
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

        # Get the POS and NER tags from spaCy
        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        ner_tags_spacy = self.nlp.get_pipe("ner").labels

        # Create a vocabulary dictionary for tags
        self.pos_tag_vocab = {tag: index + 1 for index, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: index + 1 for index, tag in enumerate(ner_tags_spacy)}

        self.input_cache = {}
        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids, additional_input=False):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        #inputs_embeds = None
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)
        # raise NotImplementedError

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = None
        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)
        # raise NotImplementedError
        # Get token type ids, since we are not considering token type,
        # this is just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        if additional_input:
            all_pos_tags = []
            all_ner_tags = []
            for sequence_id in input_ids:
                sequence_id_tup = tuple(sequence_id.tolist())
                if sequence_id_tup in self.input_cache:
                    pos_tags, ner_tags = self.input_cache[sequence_id_tup]
                else:
                    tokens = self.tokenizer.convert_ids_to_tokens(sequence_id.tolist())
                    token_strings = [token for token in tokens if token not in ["[PAD]", "[CLS]", "[SEP]"]]
                    input_string = self.tokenizer.convert_tokens_to_string(token_strings)
                    tokenized = self.nlp(input_string)
                    pos_tags = [0] * len(tokens)
                    ner_tags = [0] * len(tokens)
                    counter = -1
                    for i in range(len(token_strings)):
                        if not token_strings[i].startswith("##"):
                            counter += 1
                        pos_tags[i + 1] = self.pos_tag_vocab.get(tokenized[counter].tag_, 0)
                        ner_tags[i + 1] = self.ner_tag_vocab.get(tokenized[counter].ent_type_, 0)

                    self.input_cache[sequence_id_tup] = (pos_tags, ner_tags)

                all_pos_tags.append(pos_tags)
                all_ner_tags.append(ner_tags)

            pos_tags_ids = torch.tensor(all_pos_tags, dtype=torch.long, device=input_ids.device)
            ner_tags_ids = torch.tensor(all_ner_tags, dtype=torch.long, device=input_ids.device)
        else:
            pos_tags_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            ner_tags_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        pos_tag_embeds = self.pos_tag_embedding(pos_tags_ids)
        ner_tag_embeds = self.ner_tag_embedding(ner_tags_ids)

        embeddings = inputs_embeds + pos_embeds + tk_type_embeds + pos_tag_embeds + ner_tag_embeds
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        return embeddings

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask,additional_input=False):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids,additional_input=additional_input)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}
