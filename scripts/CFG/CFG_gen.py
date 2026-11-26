
import torch

from protgen.models.CLG_model import Transformer
from protgen.data import total_tokens, FAMILIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(num_heads=4, num_layers=2, vocabulary_size=23, embedding_size=32, n_labels=3)
model = model.to(device)

weights = torch.load(project / "results/CFG/transformer_cfg.pt", map_location=device, weights_only=True)
model.load_state_dict(weights)
model.eval()


def generate(family_id, w):

    # start with BOS
    seq = torch.tensor([[1]], device=device)
    family = torch.tensor([family_id], device=device)
    null = torch.tensor([3], device=device)

    with torch.no_grad():
        for i in range(128):

            cond = model(seq, family)
            uncond = model(seq, null)

            # last positon
            cond = cond[:, -1, :]
            uncond = uncond[:, -1, :]

            # CFG
            logits = uncond + w * (cond - uncond)

            #  avoid PAD and BOS
            logits[0,0] = -float("inf")
            logits[0,1] = -float("inf")

            probs = torch.softmax(logits, dim=1)
            next_aa = torch.multinomial(probs, 1)

            seq = torch.cat((seq, next_aa), dim=1)

            if next_aa.item() == 2:
                break

    # translate back to aminoacids
    numbers = seq[0].tolist()
    protein = ""

    for number in numbers:
        if number > 2:
            protein = protein + total_tokens[number]

    return protein

for family_id in range(3):

    print("\nFamily:", FAMILIES[family_id])

    for w in [0, 1, 2, 3, 4]:
        protein = generate(family_id, w)
        print("w =", w, ":", protein)
