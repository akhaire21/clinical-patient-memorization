"""Tiny GPT-style training and initial canary evaluation.

Run:
    python train.py --copies 0 --steps 150
    python train.py --copies 20 --steps 150
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 48
EMBEDDING_SIZE = 64
SEED = 123


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBEDDING_SIZE)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_SIZE,
            nhead=4,
            dim_feedforward=4 * EMBEDDING_SIZE,
            dropout=0.0,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.head = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, tokens):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        # Prevent the model from seeing future tokens during training.
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        x = self.layers(x, mask=mask)
        return self.head(self.norm(x))


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_from_patients(data, batch_size, pad_id, device, rng):
    inputs, targets = [], []
    for _ in range(batch_size):
        patient = rng.choice(data)
        start = rng.randint(0, max(0, len(patient) - BLOCK_SIZE - 1))
        window = patient[start:start + BLOCK_SIZE + 1]
        window = window + [pad_id] * (BLOCK_SIZE + 1 - len(window))
        inputs.append(window[:-1])
        targets.append([(-100 if token == pad_id else token) for token in window[1:]])
    return (
        torch.tensor(inputs, device=device, dtype=torch.long),
        torch.tensor(targets, device=device, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_patient(model, patient, device):
    """Score the latter half of the patient given its earlier history."""
    model.eval()
    prefix_length = max(2, len(patient) // 2)
    losses = []
    for index in range(prefix_length, len(patient)):
        context = patient[max(0, index - BLOCK_SIZE):index]
        logits = model(torch.tensor([context], device=device))[0, -1]
        losses.append(float(-F.log_softmax(logits, dim=-1)[patient[index]].item()))

    # Greedy generation of at most 20 tokens after the prefix, WITHOUT
    # feeding the true continuation back into the model.
    generated = patient[:prefix_length]
    target = patient[prefix_length:prefix_length + 20]
    for _ in target:
        context = generated[-BLOCK_SIZE:]
        logits = model(torch.tensor([context], device=device))[0, -1]
        generated.append(int(logits.argmax().item()))
    guesses = generated[prefix_length:]
    matches = sum(a == b for a, b in zip(target, guesses))
    return {
        "suffix_average_loss": round(sum(losses) / len(losses), 4),
        "first_20_greedy_token_accuracy": round(matches / len(target), 4),
        "first_20_exact_match": guesses == target,
        "number_of_scored_suffix_tokens": len(losses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, choices=[0, 20], required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")

    path = Path("data/processed.json")
    if not path.exists():
        raise FileNotFoundError("Run `python prepare.py` first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = {token: i for i, token in enumerate(data["vocab"])}
    encode = lambda seq: [token_to_id[token] for token in seq]
    train_patients = [encode(seq) for seq in data["train"]]
    canary = encode(data["canary"])
    held_out = encode(data["held_out"][0])
    # Repeating the full synthetic patient is the initial canary intervention.
    train_patients += [canary] * args.copies

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = choose_device()
    model = TinyGPT(len(data["vocab"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    rng = random.Random(SEED)
    print(f"Device: {device}; vocabulary: {len(token_to_id)}; canary copies: {args.copies}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    final_loss = None
    for step in range(1, args.steps + 1):
        x, y = batch_from_patients(train_patients, args.batch_size, token_to_id["<PAD>"], device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"Step {step:>4}/{args.steps} | training loss {final_loss:.4f}")

    result = {
        "canary_copies": args.copies,
        "steps": args.steps,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "training_loss_last_batch": round(final_loss, 4),
        "canary": evaluate_patient(model, canary, device),
        "held_out_patient": evaluate_patient(model, held_out, device),
    }
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / f"results_copies_{args.copies}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "vocab": data["vocab"]}, out / f"model_copies_{args.copies}.pt")
    print("\nResults:")
    print(json.dumps(result, indent=2))
    print(f"Saved results and model to {out}/")


if __name__ == "__main__":
    main()
