import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_token_id=None, temperature=0.8, top_k=50, top_p=0.95):
    model.eval()
    past = None
    generated = input_ids
    for _ in range(max_new_tokens):
        step_ids = generated if past is None else generated[:, -1:]
        out = model(step_ids, past_key_values=past, use_cache=True)
        token = sample_next_token(out["logits"][:, -1], temperature, top_k, top_p)
        generated = torch.cat((generated, token), dim=1)
        past = out["past_key_values"]
        if eos_token_id is not None and torch.all(token == eos_token_id):
            break
    return generated
