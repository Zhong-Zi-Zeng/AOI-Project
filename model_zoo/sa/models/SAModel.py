from transformers import SamModel


def build(args):
    sam = SamModel.from_pretrained(args['model_type'])

    for name, param in sam.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return sam
