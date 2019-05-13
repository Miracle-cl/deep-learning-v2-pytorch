import utils
from collections import Counter
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]

    return list(target_words)

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words)//batch_size

    # only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """

    # Here we're calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = (a . b) / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes

    return valid_examples, similarities

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super(SkipGramNeg, self).__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        self.inp_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        self.inp_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, x):
        inp_vectors = self.inp_embed(x)
        return inp_vectors

    def forward_output(self, x):
        out_vectors = self.out_embed(x)
        return out_vectors

    def forward_noise(self, batch_size, n_samples, device):
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist

        noise_words = torch.multinomial(noise_dist, batch_size*n_samples, replacement=True)
        noise_words = noise_words.to(device)
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, -1)
        return noise_vectors


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, inp_vectors, out_vectors, noise_vectors):
        inp_vectors = inp_vectors.unsqueeze(2) # b x ed x 1
        out_vectors = out_vectors.unsqueeze(1) # b x 1 x ed
        out_loss = torch.bmm(out_vectors, inp_vectors).sigmoid().log().squeeze() # b
        noise_loss = torch.bmm(noise_vectors.neg(), inp_vectors).sigmoid().log().squeeze() # b x n_samples
        noise_loss = noise_loss.sum(1)
        # print(out_loss.size(), noise_loss.size())
        return -(out_loss + noise_loss).mean()


def main():
    # read in the extracted text file
    text8_path = "/data/cc/dataset/text8"
    with open(text8_path) as f:
        text = f.read()

    # get list of words
    words = utils.preprocess(text)

    # print some stats about this word data
    print("Total words in text: {}".format(len(words)))
    print("Unique words: {}".format(len(set(words)))) # `set` removes any duplicate words

    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    ## subsampling
    threshold = 1e-5
    word_counts = Counter(int_words)
    #print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
    # discard some frequent words, according to the subsampling equation
    # create a new list of words for training
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    # Get our noise distribution
    # Using word frequencies calculated earlier in the notebook
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    unigram_dist = unigram_dist**(0.75)
    noise_dist = torch.from_numpy(unigram_dist / unigram_dist.sum())

    print("...............data is OK!...............")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_vocab = len(vocab_to_int)
    n_embed = 300
    batch_size = 512
    n_samples = 5
    model = SkipGramNeg(n_vocab, n_embed, noise_dist=noise_dist)
    model.to(device)

    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5
    print_every = 1500
    steps = 0
    for epoch in epochs:
        for bx, by in get_batches(train_words, batch_size):
            steps += 1
            # bx and by like : [1,2,3,4...]
            inputs = torch.LongTensor(bx).to(device)
            targets = torch.LongTensor(by).to(device)

            optimizer.zero_grad()
            inp_vectors = model.forward_input(inputs)
            out_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(batch_size, n_samples, device)

            loss = criterion(inp_vectors, out_vectors, noise_vectors)
            loss.backward()
            optimizer.step()

            # loss stats
            if steps % print_every == 0:
                print("Epoch: {}/{}".format(epoch+1, epochs))
                print("Loss: ", loss.item()) # avg batch loss at this point in training
                valid_examples, valid_similarities = cosine_similarity(model.inp_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")


if __name__ == "__main__":
    main()
