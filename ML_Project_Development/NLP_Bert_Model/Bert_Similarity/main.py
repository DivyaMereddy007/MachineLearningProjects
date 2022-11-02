#https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]
!pip3 install bert
!pip3 install torch==1.2.0 torchvision==0.4.0 -f
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

#model =SentenceTransformer(model_name)
sentence_vecs=model.encode(sentences)
sentence_vecs.shape
sentence_vecs
#768 hidden stage size

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(
    [sentence_vecs[0]],
    sentence_vecs[1:]

)


####### Same code with transformers and torch
model_name='sentence-transformers/bert-base-nli-mean-tockens'

from transformers import AutoTokenizer, AutoModel
import torch
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)

tokens={'input_ids':[],'attention_mask':[]}
for sentence in sentences:
        new_tockens=tokenizer.encode_plus(sentence,max_len=128,truncation=True, padding='max_length',return_tensors='pt')
        new_tockens['input_ids'].append(new_tockens['input_ids'])
        tockens['attention_mask'].append(new_tokens['attention_mask'][0])


outputs=model(**tockens)

outputs.keys()

embeddings=outpurs.last_hidden_state
embeddings.shape


In [7]:
attention_mask = tokens['attention_mask']
attention_mask.shape


mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
mask.shape

mask

masked_embeddings = embeddings * mask
masked_embeddings.shape

masked_embeddings

summed = torch.sum(masked_embeddings, 1)
summed.shape

summed_mask = torch.clamp(mask.sum(1), min=1e-9)
summed_mask.shape

summed_mask

mean_pooled = summed / summed_mask

mean_pooled
