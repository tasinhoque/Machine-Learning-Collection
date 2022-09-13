import spacy
from torchtext.data import BucketIterator, Field, TabularDataset

spacy_en = spacy.load("en_core_web_sm")


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)
fields = {"quote": ("q", quote), "score": ("s", score)}


train_data, test_data = TabularDataset.splits(
    path="mydata", train="train.json", test="test.json", format="json", fields=fields
)

# train_data, test_data = TabularDataset.splits(
#     path="mydata", train="train.csv", test="test.csv", format="csv", fields=fields
# )

# train_data, test_data = TabularDataset.splits(
#     path="mydata", train="train.tsv", test="test.tsv", format="tsv", fields=fields
# )

quote.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=2, device="cuda"
)

for batch in train_iterator:
    print(batch.q)
    print(batch.s)
