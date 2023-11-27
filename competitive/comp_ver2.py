import numpy as np
import torch.nn as nn
import torch
import re
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging

from datasets import load_dataset
from transformers import NougatProcessor,VisionEncoderDecoderModel
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel,PretrainedConfig
from transformers import SwinConfig, SwinModel
from timm.models import SwinTransformer
from timm import create_model
from torchvision.transforms.functional import resize, rotate
from PIL import ImageOps
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast, StoppingCriteria, StoppingCriteriaList, MBartConfig, MBartForCausalLM, MBartTokenizerFast
from collections import defaultdict

#idea -> encode the image using a swin transformer, then decode using a transformer decoder

class SwinEncoder(nn.Module):
    def __init__(self,input_size,window_size,encoder_layer,patch_size,emb_dim,num_heads,name_or_path = None):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.name_or_path = name_or_path
        # self.config = SwinConfig(image_size=224,patch_size=self.patch_size,num_channels = 3,embed_dim=self.emb_dim,depth=12,num_heads=self.num_heads,mlp_ratio=4,qkv_bias=True)
        # self.model = SwinModel(self.config)
        self.model = SwinTransformer(
            img_size=self.input_size,
            patch_size=self.patch_size,
            depths=self.encoder_layer,
            num_heads=self.num_heads,
            embed_dim=self.emb_dim,
            window_size=self.window_size,
            num_classes=0)
        
        if not name_or_path:
            swin_state_dict = create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(np.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)
        

    def forward(self,x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        # x = self.model.norm(x) -> to look
        return x
    
    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @property
    def to_tensor(self):
        pass
        # if self.training:
        #     !return train_transform
        # else:
        #     !return test_transform

    def prepare_input(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))
    

class BARTDecoder(nn.Module):
    def __init__(self,decoder_layer,max_pos_emb,hidden_dim = 1024,path = None):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_pos_emb = max_pos_emb
        self.hidden_dim = hidden_dim
        if not path:
            tokenizer_file = Path(__file__).parent / "dataset" / "tokenizer.json"
        else:
            tokenizer_file = Path(path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise ValueError("Could not find tokenizer file")
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file)) #uses a pretrained tokenizer
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        config = MBartConfig(vocab_size=len(self.tokenizer),is_decoder=True,is_encoder_decoder=False,add_cross_attention=True, scale_embedding=True
                             ,decoder_layers=self.decoder_layer,d_model=self.hidden_dim,max_position_embeddings=self.max_pos_emb,add_final_layer_norm = True)
        
        self.model = MBartForCausalLM(config)
        self.model.config.is_encoder_decoder = True
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        if not path:
            bart_state_dict = MBartForCausalLM.from_pretrained(
                "facebook/mbart-large-50"
            ).state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if (
                    x.endswith("embed_positions.weight")
                    and self.max_position_embeddings != 1024
                ):
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][
                        : len(self.tokenizer), :
                    ]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict, strict=False)

    def add_new_tokens(self, new_tokens: list):
        """
        Add new tokens to the tokenizer and the model's embedding layer.
        """
        newly_added_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(new_tokens))})
        if(newly_added_tokens>0):
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(
            self,
            input_ids: torch.Tensor,
            encoder_outputs: torch.Tensor,
            past=None,
            past_key_values=None,
            use_cache: bool = None,
            attention_mask: torch.Tensor = None,
        ):
        # """
        # Args:
        #     input_ids: (batch_size, sequence_length)

        # Returns:
        #     input_ids: (batch_size, sequence_length)
        #     attention_mask: (batch_size, sequence_length)
        #     encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        # """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        past = past or past_key_values
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output
    
    def forward(self,input_ids): #check attention mask and stuff parameters later when necessary
        return self.model.forward(input_ids)
    
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of MBart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight
    
class LatexModelConfig(PretrainedConfig):
    def __init__(self,input_size: list[int] = [224,224],window_size = 7,encoder_layer:list[int] = [2,2,6,2],decoder_layer:int = 10,
                 max_pos_emb:int = None,max_length:int = 4096, path = "",patch_size = 4, num_heads = [4,8,16,32],hid_dim = 1024, emb_dim = 128, **kwargs,):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_pos_emb = (max_length if max_pos_emb == None else max_pos_emb)
        self.patch_size = patch_size
        self.max_length = max_length
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        self.path = path
        self.emb_dim = emb_dim


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


def batch(l, b=15):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[i : i + b])
    return subs


def subdiv(l, b=10):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[: i + b])
    return subs


class LatexModel(PreTrainedModel):
    config_class = LatexModelConfig

    def __init__(self,config:LatexModelConfig):
        self.config = config
        self.encoder = SwinEncoder(input_size = self.config.input_size,window_size = self.config.window_size,encoder_layer = self.config.encoder_layer,patch_size = self.config.patch_size,emb_dim = self.config.emb_dim,num_heads = self.config.num_heads)
        self.decoder = BARTDecoder(decoder_layer = self.config.decoder_layer,max_pos_emb = self.config.max_pos_emb,hidden_dim = self.config.hid_dim,path = self.config.path)

    def forward(self,image_tensors,decoder_inputs,attention_mask = None):
        encoder_outputs = self.encoder(image_tensors)
        outputs = self.decoder(input_ids = decoder_inputs[:,:-1].contiguous(),
                               encoder_hidden_states = encoder_outputs,
                               attention_mask=attention_mask[:, :-1],
                               labels=decoder_inputs[:, 1:].contiguous(),


        )
        return outputs
    
    def _init_weights(self, *args, **kwargs):
        return

    def inference(
        self,
        image: Image.Image = None,
        image_tensors = None,
        return_attentions: bool = False,
        early_stopping: bool = True,
    ):
        """
        Generate a token sequence in an auto-regressive manner.

        Args:
            image: input document image (PIL.Image)
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
        """
        output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
        }
        if image is None and image_tensors is None:
            logging.warn("Image not found")
            return output

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type != "mps":
            image_tensors = image_tensors.to(next(self.parameters()).dtype)

        image_tensors = image_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)

        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden_state, attentions=None
        )

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.unsqueeze(0)
            )

        # get decoder output
        decoder_output = self.decoder.model.generate(
            encoder_outputs=encoder_outputs,
            min_length=1,
            max_length=self.config.max_length,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[
                [self.decoder.tokenizer.unk_token_id],
            ],
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=return_attentions,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList(
                [StoppingCriteriaScores()] if early_stopping else []
            ),
        )
        output["repetitions"] = decoder_output.sequences.clone()
        output["sequences"] = decoder_output.sequences.clone()
        batch_size = len(decoder_output.sequences)

        logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)
        values = logits.values
        indices = logits.indices

        for b in range(batch_size):
            mask = indices[b] != self.decoder.tokenizer.pad_token_id
            N = mask.sum().item()
            var = np.array(
                [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
            )
            if len(var) < 10:
                output["repeats"].append(None)
                continue
            varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1])
            minlen = 120
            if (
                indices[b] == self.decoder.tokenizer.eos_token_id
            ).any() and N + 1 < indices.shape[1]:
                # there is an end to the generation, likely no repetitions
                output["repeats"].append(None)
                continue
            small_var = np.where(varvar < 0.045)[0]
            if early_stopping and len(small_var) > 1:
                if np.all(np.diff(small_var) < 2):
                    idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                    if idx / N > 0.9:  # at most last bit
                        output["repeats"].append(None)
                        continue
                    elif small_var[0] < 30:
                        idx = 0
                    logging.warn("Found repetitions in sample %i" % b)
                    output["repeats"].append(idx)
                    output["sequences"][b, idx:] = self.decoder.tokenizer.pad_token_id
                    output["repetitions"][b, :idx] = self.decoder.tokenizer.pad_token_id
                else:
                    output["repeats"].append(None)
            else:
                output["repeats"].append(None)
        output["repetitions"] = self.decoder.tokenizer.batch_decode(
            output["repetitions"], skip_special_tokens=True
        )
        output["predictions"] = !postprocess(
            self.decoder.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            ),
            markdown_fix=False,
        )

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained nougat model from a pre-trained model configuration

        Args:
            model_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local.
        """
        model = super(LatexModel, cls).from_pretrained(
            model_path, *model_args, **kwargs
        )

        # truncate or interpolate position embeddings of decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model

                             