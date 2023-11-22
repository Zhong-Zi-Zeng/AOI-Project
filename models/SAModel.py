from transformers import SamModel


def build(args):
    model = SamModel.from_pretrained(args['pretrained_model'])

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model
