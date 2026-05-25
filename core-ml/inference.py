import sys
import time

import torch as t
from transformers import GPT2Tokenizer

from transformer_blocks import ModularTransformer
from attention_heads import MaskedMultiHeadedAttention, FlexibleAttentionBlock


# -------------------------------------------------------------
# Rebuild the model from a checkpoint config
# (attention blocks are built outside ModularTransformer)
# -------------------------------------------------------------
def build_attention(mcfg, ctx_len, dim, heads):
    a = mcfg.get("attention", {"type": "mha"})
    if a.get("type", "mha") == "mha":
        return MaskedMultiHeadedAttention(heads, ctx_len, dim)

    kw = {}
    if a["variant"] == "SWA":
        kw["window_size"] = a["window_size"]
    if a["variant"] == "GQA":
        kw["groups"] = a["groups"]
    if a["pe"] == "RPE":
        kw["max_distance"] = a["max_distance"]

    return FlexibleAttentionBlock(
        pe=a["pe"], variant=a["variant"], dim=dim, seq_len=ctx_len, heads=heads, **kw
    )


def build_model(cfg, vocab_size, device):
    m = cfg["model"]
    ctx_len = m["context_len"]
    dim = m["embed_dim"]
    heads = m["n_heads"]
    n_layers = m["n_layers"]
    mode = m.get("mode", "baseline")

    n_attn = n_layers // 2 if mode == "alternating" else n_layers
    attention_blocks = [
        build_attention(m, ctx_len, dim, heads) for _ in range(n_attn)
    ]

    conv = m.get("conv", {"kernel_size": 3, "padding": 1})

    return ModularTransformer(
        ctx_len=ctx_len,
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        attention_blocks=attention_blocks,
        mode=mode,
        conv_cfg={"kernel_size": conv["kernel_size"], "padding": conv["padding"]},
        use_abs_pe=m.get("use_abs_pe", True),
    ).to(device)


def load_model(checkpoint_path, device="cuda"):
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    vocab_size = checkpoint["vocab_size"]

    model = build_model(cfg, vocab_size, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


# -------------------------------------------------------------
# Autoregressive generation with optional token streaming
# returns (full_text, stats) where stats carries throughput
# -------------------------------------------------------------
@t.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    ctx_len=1024,
    stream=True,
):
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_token_id]
    tokens = t.tensor([ids], dtype=t.long, device=device)

    if stream:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    temperature = max(temperature, 1e-6)
    pieces = []
    n_generated = 0
    first_token_time = None
    start = time.perf_counter()

    for _ in range(max_new_tokens):
        input_tokens = tokens[:, -ctx_len:]  # crop to context length

        logits = model(input_tokens)[:, -1, :] / temperature  # last position only

        if top_k > 0:
            values, _ = t.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = t.softmax(logits, dim=-1)
        next_token = t.multinomial(probs, num_samples=1)

        if first_token_time is None:
            first_token_time = time.perf_counter()

        tokens = t.cat([tokens, next_token], dim=1)
        n_generated += 1

        piece = tokenizer.decode(next_token[0].tolist())
        pieces.append(piece)
        if stream:
            sys.stdout.write(piece)
            sys.stdout.flush()

        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else 0.0
    # decode throughput excludes prefill / first-token latency
    decode_time = max(elapsed - ttft, 1e-9)

    stats = {
        "tokens": n_generated,
        "elapsed": elapsed,
        "ttft": ttft,
        "tok_s": n_generated / max(elapsed, 1e-9),
        "decode_tok_s": (n_generated - 1) / decode_time if n_generated > 1 else 0.0,
    }
    return prompt + "".join(pieces), stats


def print_stats(stats):
    print(
        f"\n── {stats['tokens']} tokens in {stats['elapsed']:.2f}s | "
        f"{stats['tok_s']:.1f} tok/s overall | "
        f"{stats['decode_tok_s']:.1f} tok/s decode | "
        f"TTFT {stats['ttft'] * 1000:.0f}ms ──"
    )


def interactive_loop(model, tokenizer, ctx_len, args):
    print("\nInteractive generation. Type a prompt, empty line or Ctrl-D to quit.")
    while True:
        try:
            prompt = input("\nprompt> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        _, stats = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            ctx_len=ctx_len,
            stream=True,
        )
        print_stats(stats)
    print("bye.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top-k filtering")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--interactive", action="store_true", help="prompt loop instead of single prompt")
    parser.add_argument("--no_stream", action="store_true", help="disable token streaming")
    args = parser.parse_args()

    device = t.device(args.device if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model, cfg = load_model(args.checkpoint, device=device)
    ctx_len = cfg["model"]["context_len"]
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if args.interactive:
        interactive_loop(model, tokenizer, ctx_len, args)
    else:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        _, stats = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            ctx_len=ctx_len,
            stream=not args.no_stream,
        )
        print_stats(stats)
