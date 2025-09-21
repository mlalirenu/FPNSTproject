import torch
import math


def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Content Layer
            "28": "conv5_1"
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features



def compute_content_loss(vgg, content_img, generated_img, device=torch.device("cpu")):
    content_features = get_features(content_img, vgg)
    generated_features = get_features(generated_img, vgg)
    content_loss =  torch.mean((generated_features["conv4_2"] - content_features["conv4_2"]) ** 2)
    content_similarity = 100 * math.exp(-content_loss.item() / 10)
    return content_loss, content_similarity


def gram_matrix(tensor):
    if tensor.dim() == 3:
        # [C, H, W] → add batch dimension
        tensor = tensor.unsqueeze(0)
    b, d, h, w = tensor.size()
    features = tensor.view(b * d, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * d * h * w)


def compute_style_loss(vgg, style_img, generated_img, device=torch.device("cpu")):
    style_features = get_features(style_img, vgg)
    generated_features = get_features(generated_img, vgg)

    style_loss = 0
    style_weights = {
        "conv1_1": 0.2,
        "conv2_1": 0.2,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    for layer in style_weights:
        gen_feat = generated_features[layer]
        style_feat = style_features[layer]

        G = gram_matrix(gen_feat.squeeze(0))
        A = gram_matrix(style_feat.squeeze(0))

        d, m = gen_feat.size(1), gen_feat.size(2) * gen_feat.size(3)
        layer_style_loss = torch.sum((G - A) ** 2) / (4 * (d**2) * (m**2))

        style_loss += style_weights[layer] * layer_style_loss

    style_similarity = 100 * math.exp(-style_loss.item() / 10)
    return style_loss, style_similarity
