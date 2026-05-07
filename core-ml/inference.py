import torch as t
from transformers import GPT2Tokenizer

from transformer_blocks import ModularTransformer


def load_model(checkpoint_path, device="cuda"):
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    vocab_size = checkpoint["vocab_size"]

    model = ModularTransformer(
        ctx_len=cfg["model"]["context_len"],
        dim=cfg["model"]["embed_dim"],
        heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        vocab_size=vocab_size,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


@t.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, ctx_len=1024):
    tokens = tokenizer.encode(prompt)
    tokens = t.tensor([tokens], dtype=t.long, device=next(model.parameters()).device)

    for _ in range(max_new_tokens):
        # crop to context length if needed
        input_tokens = tokens[:, -ctx_len:]

        logits = model(input_tokens)  # (B, L, vocab_size)
        logits = logits[:, -1, :] / temperature  # only care about the last position

        # top-k filtering
        if top_k > 0:
            values, _ = t.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = t.softmax(logits, dim=-1)
        next_token = t.multinomial(probs, num_samples=1)

        tokens = t.cat([tokens, next_token], dim=1)

        # stop if we hit eos
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top-k filtering")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    args = parser.parse_args()

    device = t.device(args.device if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model, cfg = load_model(args.checkpoint, device=device)
    ctx_len = cfg["model"]["context_len"]
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        ctx_len=ctx_len,
    )
    print(output)
