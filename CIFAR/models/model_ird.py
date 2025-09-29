import sys
sys.path.insert(0, 'models')

from wrn import *

from clip_utils import clip
from PIL import Image
from clip_utils.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names

        n_cls = len(classnames)
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        # print('coooooooooooop image_features.shape', image_features.shape)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print('image_features.shape', image_features.shape)
        # print('text_features.shape', text_features.shape)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # print('logit_scale', logit_scale)

        return logits, image_features


class IRD(nn.Module):
    """
    OoD-Aware alignment learning by max-margin between ood and id data
    """

    def __init__(self, num_classes, classnames):
        super(IRD, self).__init__()
        self.num_classes = num_classes
        self.classnames = classnames  # list(classnames.keys())
        self.n_cls = len(classnames)
        self.name_lens = [name for name in self.classnames]
        print("classnames:", self.classnames)
        self.n_ctx = 16
        self.ctx_dim = 512

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        # self.dtype = self.clip_model.dtype

        print("Building custom CLIP")
        self.model = CustomCLIP(self.classnames, self.clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.theta = nn.Parameter(torch.tensor(1.))
        self.theta.requires_grad_(True)

    def mm_energy(self, src_logits, image_feature, label_id):
        energy = -torch.logsumexp(src_logits, dim=1)

        id = label_id == 0
        ood = label_id == 1

        # SC = src_logits.sigmoid().max(dim=1)[0]
        loss_id = -(torch.log(1 - (self.theta * energy[id]).sigmoid())).mean()

        # ############################## cosine similarity for minimizing mutual information #################################
        S = (image_feature @ image_feature.transpose(-1, -2)).sigmoid()  # self-similarity
        mask = (1. * ood).unsqueeze(1) @ (1. * id).unsqueeze(1).T
        mutual_info = (S * mask).sum(dim=1)
        # ##########################################################################################

        mutual_info = mutual_info / (1e-6 + mm_weight.norm(dim=-1, keepdim=True))
        loss_ood = -(torch.log((self.theta * energy[ood]).sigmoid())).mean()

        loss_mm = loss_id + loss_ood + mutual_info.mean()
        return loss_mm

    def forward(self, x):

        x_class, image_features = self.model(x)

        x_ood = torch.sigmoid(-self.theta * torch.logsumexp(x_class, dim=-1)).unsqueeze(-1)

        x_all = torch.cat([x_class, x_ood], dim=1)  # class label and id-ood label

        return x_all, image_features




