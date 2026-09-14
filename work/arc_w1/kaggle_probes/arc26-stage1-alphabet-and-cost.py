"""Stage 1: pin down the 16-token alphabet, the serialisation format, and the real
throughput / backward-pass cost on T4 x2.

Stage 0 established (from the artifact itself):
    Qwen3ForCausalLM, hidden 2560, 36 layers, vocab_size = 16, params 3.634B
    ids 13 = <|endoftext|> (pad), 14 = <|im_start|>, 15 = <|im_end|> (eos), no chat template
    => ids 0..12 are the 13 grid symbols. This stage dumps them and probes encoding.

Every block is wrapped so one failure cannot hide the others.
"""
import json
import os
import time
import traceback

import torch

MODEL_DIR = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"


def block(title):
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8, flush=True)


block("alphabet: raw tokenizer files")
for name in ("vocab.json", "added_tokens.json"):
    fp = os.path.join(MODEL_DIR, name)
    if os.path.exists(fp):
        try:
            raw = open(fp, encoding="utf-8").read()
            print(f"--- {name} ({len(raw)} bytes) ---")
            print(raw[:3000])
        except Exception:
            traceback.print_exc()

block("tokenizer.json: model.vocab")
try:
    tj = json.load(open(os.path.join(MODEL_DIR, "tokenizer.json"), encoding="utf-8"))
    print("model.type:", tj.get("model", {}).get("type"))
    print("model.vocab:", json.dumps(tj.get("model", {}).get("vocab"), ensure_ascii=False)[:2000])
    print("model.merges:", json.dumps(tj.get("model", {}).get("merges"))[:800])
    print("added_tokens:", json.dumps(tj.get("added_tokens"), ensure_ascii=False)[:1500])
    print("pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"))[:800])
    print("decoder:", json.dumps(tj.get("decoder"))[:500])
except Exception:
    traceback.print_exc()

block("encoding round-trips")
try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    probes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "0123456789",
        "0\n1", "01\n23", "0 1",
        "012;345", "012|345", "012,345",
        "[[0,1],[2,3]]",
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "0\n1\n\n2\n3",
    ]
    for p in probes:
        try:
            ids = tok.encode(p, add_special_tokens=False)
            back = tok.decode(ids)
            print(f"  {p!r:24} -> {ids}   decode-> {back!r}")
        except Exception as exc:
            print(f"  {p!r:24} -> FAILED {type(exc).__name__}: {exc}")
    print("all_special_ids:", tok.all_special_ids)
    print("id->token map:", {i: tok.convert_ids_to_tokens(i) for i in range(16)})
except Exception:
    traceback.print_exc()

block("model load + forward/backward timing")
try:
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print("load_seconds:", round(time.time() - t0, 1))
    print("params_B:", round(sum(p.numel() for p in model.parameters()) / 1e9, 3))

    VOCAB = 16
    for seq in (512, 1024, 2048, 4096):
        try:
            ids = torch.randint(0, VOCAB, (1, seq), device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model(input_ids=ids)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"  forward  seq={seq:<5} {dt * 1000:8.1f} ms  ({seq / dt:7.0f} tok/s)")
        except Exception as exc:
            print(f"  forward seq={seq} FAILED {type(exc).__name__}: {exc}")
            break

    del model
    torch.cuda.empty_cache()
except Exception:
    traceback.print_exc()

block("LoRA TTT step cost (the real budget driver)")
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=dtype, device_map="cuda:0", trust_remote_code=True
        )
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        m = get_peft_model(m, cfg)
        m.train()
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{dtype_name}] wrapped in {time.time() - t0:.1f}s, trainable={trainable / 1e6:.2f}M")
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        for seq in (1024, 2048):
            for accum in (1, 4):
                try:
                    ids = torch.randint(0, 16, (accum, seq), device="cuda")
                    torch.cuda.synchronize()
                    t0 = time.time()
                    out = m(input_ids=ids, labels=ids)
                    out.loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    toks = accum * seq
                    print(f"  [{dtype_name}] step seq={seq:<5} batch={accum}  "
                          f"{dt * 1000:8.1f} ms  ({toks / dt:7.0f} tok/s)  loss={out.loss.item():.3f}")
                except Exception as exc:
                    print(f"  [{dtype_name}] step seq={seq} batch={accum} FAILED "
                          f"{type(exc).__name__}: {exc}")
                    torch.cuda.empty_cache()
        del m, opt
        torch.cuda.empty_cache()
except Exception:
    traceback.print_exc()

block("memory")
try:
    free, total = torch.cuda.mem_get_info()
    print(f"free {free / 2**30:.1f} GiB / total {total / 2**30:.1f} GiB")
except Exception:
    traceback.print_exc()

print("\nstage1 complete", flush=True)
