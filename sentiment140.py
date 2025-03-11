from datasets import load_dataset
import tiktoken
import os
import torch

# Load train dataset 
ds = load_dataset("stanfordnlp/sentiment140", split="train", trust_remote_code=True)
ds = ds.shuffle(seed=1337)

# Load test dataset 
test_ds = load_dataset("mteb/tweet_sentiment_extraction", split="test")

enc = tiktoken.get_encoding("gpt2")

# ----------------------------------------------------------------------------

# Calculate average tokens per tweet:

# num_tweets = len(ds["text"])
# mean = 0
# for tweet in ds["text"]:
#     mean += len(enc.encode_ordinary(tweet)) / num_tweets

# print(mean) # Average tweet is ~18 tokens, so use a block size of 16

# ----------------------------------------------------------------------------

# Create a "tweets" folder and save tweets and their labels as .pt:

local_dir = "sentiment140"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Create padded arrays of tokenized tweets:

eot = enc._special_tokens['<|endoftext|>'] # End of text token
enc._special_tokens['<|pad|>'] = eot + 1
pad = enc._special_tokens['<|pad|>'] # pad token

# tweets = []
# labels = []

# Tweets:

for tweet in ds["text"]:
    tokens = enc.encode_ordinary(tweet)

    # Pad or truncate tweets so they're exactly 16 tokens:

    if len(tokens) < 15: # Pad with token 0 if shorter than 15
        enc_tweet = [eot] + tokens + [pad] * (15 - len(tokens))
    else: # Truncate if longer than 15
        enc_tweet = [eot] + tokens[:15]

    tweets += enc_tweet
    
# Save tweets:

tweets_tensor = torch.tensor(tweets, dtype=torch.long)
tweets_path = os.path.join(DATA_CACHE_DIR, "tweets.pt")
torch.save(tweets_tensor, tweets_path)

# Labels:

for label in ds["sentiment"]:
    if label == 4:
            labels += [1]  # Positive
    else:
        labels += [0] # Negative

# Save labels:

labels_tensor = torch.tensor(labels, dtype=torch.long)
labels_path = os.path.join(DATA_CACHE_DIR, "labels.pt")
torch.save(labels_tensor, labels_path)

# ----------------------------------------------------------------------------

# save test tweets and their labels as .pt:

# Create padded arrays of tokenized tweets:

tweets = []
labels = []

# Tweets:

for tweet, sentiment in zip(test_ds["text"], test_ds["label"]):
    if sentiment != 1:
        tokens = enc.encode_ordinary(tweet)
        
        # Pad or truncate tweets so they're exactly 16 tokens:

        if len(tokens) < 15:
            enc_tweet = [eot] + tokens + [pad] * (15 - len(tokens))
        else:
            enc_tweet = [eot] + tokens[:15]
        
        tweets += enc_tweet
        
        if sentiment == 2:
            labels += [1] # Positive
        else:
            labels += [0] # Negative

# Save test tweets and labels:

tweets_tensor = torch.tensor(tweets, dtype=torch.long)
tweets_path = os.path.join(DATA_CACHE_DIR, "test_tweets.pt")
torch.save(tweets_tensor, tweets_path)

labels_tensor = torch.tensor(labels, dtype=torch.long)
labels_path = os.path.join(DATA_CACHE_DIR, "test_labels.pt")
torch.save(labels_tensor, labels_path)