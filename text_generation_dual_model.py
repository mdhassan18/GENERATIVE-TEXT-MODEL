import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------
# GPT-2 TEXT GENERATION
# ---------------------------
def generate_with_gpt2(prompt, max_length=100):
    print("🔁 Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# ---------------------------
# BASIC LSTM TEXT GENERATION
# ---------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, layer_dim):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

def train_dummy_lstm(text, n_epochs=10):
    print("🔧 Training basic LSTM on dummy text...")
    chars = list(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    vocab_size = len(chars)

    data = [char2idx[ch] for ch in text]
    seq_length = 20
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])

    X = torch.tensor(inputs)
    y = torch.tensor(targets)

    model = CharLSTM(vocab_size, 128, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        hidden = None
        output, hidden = model(X, hidden)
        loss = loss_fn(output, y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    return model, char2idx, idx2char

def generate_lstm_text(model, char2idx, idx2char, seed, length=200):
    model.eval()
    input_seq = [char2idx[ch] for ch in seed[-20:] if ch in char2idx]
    input_tensor = torch.tensor(input_seq).unsqueeze(0)
    hidden = None
    result = seed
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            last_char_logits = output[-1]
            probs = F.softmax(last_char_logits, dim=0).cpu().numpy()
            next_char_idx = random.choices(range(len(probs)), weights=probs)[0]
            next_char = idx2char[next_char_idx]
            result += next_char
            input_tensor = torch.tensor([[next_char_idx]])
    return result

# ---------------------------
# MAIN INTERFACE
# ---------------------------
def main():
    print("\n=== TEXT GENERATION MODEL ===")
    print("1. GPT-2 (Pre-trained, better output)")
    print("2. LSTM (Train locally, basic output)")
    choice = input("Select model (1 or 2): ").strip()

    if choice == '1':
        prompt = input("Enter a prompt to begin: ")
        generated = generate_with_gpt2(prompt)
        print("\n📝 Generated Text:\n", generated)

    elif choice == '2':
        print("\nUsing LSTM on sample text...")
        sample_text = "hello world this is a tiny lstm text generator for demo only "
        model, char2idx, idx2char = train_dummy_lstm(sample_text * 100, n_epochs=10)
        seed = input("Enter a starting seed (a few words): ")
        generated = generate_lstm_text(model, char2idx, idx2char, seed)
        print("\n📝 Generated Text:\n", generated)

    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
