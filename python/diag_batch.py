import torch, random, sys

its = torch.load("data/train_items.pt", weights_only=False)
order = list(range(len(its)))
random.Random(20260922).shuffle(order)
train = [its[i] for i in sorted(order[400:])]
random.seed(42 + 0 + 0)
random.shuffle(train)
for b in range(8016, 8112, 8):
    chunk = train[b : b + 8]
    lens = [len(c["ids"]) for c in chunk]
    labels = [c["label"] for c in chunk]
    print("b_idx=%d lens=%s labels=%s" % (b, lens, labels))
