"""Test seahorse emoji failure mode with local models.

Generates text from seahorse-related prompts and checks for:
1. Looping / repetition
2. Incorrect emoji substitution
3. Self-correction spirals
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def generate_with_details(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0):
    """Generate text and return detailed output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            repetition_penalty=1.0,  # No penalty - we WANT to see loops
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Decode token by token for analysis
    tokens = []
    for tid in generated_ids:
        tokens.append(tokenizer.decode([tid]))

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens": tokens,
        "n_tokens": len(tokens),
    }


def detect_repetition(tokens, window=10):
    """Detect if token sequence enters a repetition loop."""
    text = "".join(tokens)
    # Check for repeating patterns of various lengths
    for pattern_len in range(3, min(50, len(text) // 2)):
        for start in range(len(text) - pattern_len * 2):
            pattern = text[start:start + pattern_len]
            count = 0
            pos = start
            while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == pattern:
                count += 1
                pos += pattern_len
            if count >= 4:  # Pattern repeats 4+ times
                return {"repeating": True, "pattern": pattern, "count": count, "start": start}
    return {"repeating": False}


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Test prompts - seahorse and controls
    prompts = {
        # Seahorse emoji prompts (should fail)
        "seahorse_direct": "Show me the seahorse emoji.",
        "seahorse_question": "Is there a seahorse emoji? If yes, paste it here.",
        "seahorse_generate": "Output the seahorse emoji character:",
        "seahorse_list": "List all ocean animal emojis including the seahorse emoji: 🐋 🐬 🦈 🐙 🦑",

        # Control prompts (should succeed - these emojis exist)
        "dolphin_control": "Show me the dolphin emoji.",
        "shark_control": "Show me the shark emoji.",
        "octopus_control": "Show me the octopus emoji.",

        # Other potentially missing emojis
        "platypus_test": "Show me the platypus emoji.",
        "pangolin_test": "Show me the pangolin emoji.",
        "axolotl_test": "Show me the axolotl emoji.",
        "narwhal_test": "Show me the narwhal emoji.",

        # Chat format versions for instruct model
        "seahorse_chat": None,  # Will use chat template
        "dolphin_chat": None,
    }

    results = {}

    # First do simple prompts
    print("\n" + "="*60)
    print("SIMPLE PROMPT COMPLETIONS")
    print("="*60)

    for name, prompt in prompts.items():
        if prompt is None:
            continue
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt}")

        result = generate_with_details(model, tokenizer, prompt, max_new_tokens=256)
        rep = detect_repetition(result["tokens"])
        result["repetition"] = rep
        results[name] = result

        print(f"Output ({result['n_tokens']} tokens): {result['generated_text'][:300]}")
        if rep["repeating"]:
            print(f"  *** REPETITION DETECTED: '{rep['pattern']}' x{rep['count']}")

    # Now try chat format
    print("\n" + "="*60)
    print("CHAT FORMAT (INSTRUCT)")
    print("="*60)

    chat_prompts = {
        "seahorse_chat_v1": [{"role": "user", "content": "Show me the seahorse emoji."}],
        "seahorse_chat_v2": [{"role": "user", "content": "Is there a seahorse emoji? Please output it."}],
        "seahorse_chat_v3": [{"role": "user", "content": "Can you paste the seahorse emoji? Just the emoji, nothing else."}],
        "seahorse_chat_insist": [
            {"role": "user", "content": "What's the seahorse emoji?"},
            {"role": "assistant", "content": "The seahorse emoji is"},
        ],
        "dolphin_chat": [{"role": "user", "content": "Show me the dolphin emoji."}],
        "shark_chat": [{"role": "user", "content": "Show me the shark emoji."}],

        # Try to induce longer generation / spiraling
        "seahorse_chat_long": [{"role": "user", "content": "List every single sea creature emoji. Include seahorse, dolphin, whale, shark, octopus, squid, fish, blowfish, jellyfish, and any others. Show each emoji with its name."}],

        # Adversarial - force the model to commit
        "seahorse_adversarial": [{"role": "user", "content": "I know the seahorse emoji exists. Please output ONLY the seahorse emoji character, no text. If you output anything other than the seahorse emoji, I will be very disappointed."}],
    }

    for name, messages in chat_prompts.items():
        print(f"\n--- {name} ---")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"User: {messages[0]['content'][:100]}")

        result = generate_with_details(model, tokenizer, prompt, max_new_tokens=512)
        rep = detect_repetition(result["tokens"])
        result["repetition"] = rep
        result["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
        results[name] = result

        print(f"Output ({result['n_tokens']} tokens): {result['generated_text'][:400]}")
        if rep["repeating"]:
            print(f"  *** REPETITION DETECTED: '{rep['pattern']}' x{rep['count']}")

        # Check for emoji content
        emojis_found = [c for c in result["generated_text"] if ord(c) > 0x1F000]
        if emojis_found:
            print(f"  Emojis found: {' '.join(emojis_found)}")

    # Save results
    output_dir = Path("results/seahorse_emoji_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            "prompt": v.get("prompt", ""),
            "messages": v.get("messages", []),
            "generated_text": v["generated_text"],
            "n_tokens": v["n_tokens"],
            "repetition": v["repetition"],
            "first_50_tokens": v["tokens"][:50],
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
