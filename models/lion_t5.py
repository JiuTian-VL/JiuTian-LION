import contextlib
import logging
import string
from typing import Literal, Union, List
from PIL.Image import Image

import torch
import torch.nn as nn
from icecream import ic
from torch.cuda.amp import autocast as autocast
from transformers import BertTokenizer, T5TokenizerFast
from transformers.modeling_outputs import BaseModelOutput

from common.registry import registry
from models.base_model import BaseModel
from models.eva_vit import create_eva_vit_g
from models.lion_adapters import FusionAdapter, set_adapter_t5, set_router_idx
from models.modeling_t5 import T5Config, T5ForConditionalGeneration
from models.Qformer import BertConfig, BertLMHeadModel
from ram import get_transform
from ram.models import ram


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@registry.register_model("lion_t5")
class LIONT5InstructAdapter(BaseModel):
    """
    LION T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from models import load_model
        >>> model = load_model("lion_t5", "flant5xl")
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/lion_flant5xl.yaml",
        "flant5xxl": "configs/models/lion_flant5xxl.yaml",
    }

    def __init__(
        self,
        bert_model,
        vit_model,
        llm_model,
        ram_model,
        max_txt_len=128,
        max_output_txt_len=128,
        visual_input: Literal["ALL", "QFORMER", "AGGREGATOR"] = "ALL",
        enable_semantic_tags=True,
        boost_lr_scale=1,

        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
    ):
        super().__init__()
        assert bert_model is not None, "The path for bert model is not provided."
        assert vit_model is not None, "The path for vit model is not provided."
        assert llm_model is not None, "The path for llm model is not provided."
        assert visual_input in ["ALL", "QFORMER", "AGGREGATOR"], f"Invalid visual input type: {visual_input}."
        self.bert_model = bert_model
        self.visual_input = visual_input
        self.enable_semantic_tags = enable_semantic_tags
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.boost_lr_scale = boost_lr_scale
        self.ram_path = ram_model
        self.ram_model = None
        logging.info(f"visual_input: {visual_input}")

        print("Loading VIT")
        self.visual_encoder = self._init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")
        
        self._init_llm(llm_model)
        
        if self.visual_input != "AGGREGATOR":
            print("Loading QFormer")
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, truncation_side="left")
            self.bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.Qformer, self.query_tokens = self._init_Qformer(
                bert_model, num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.resize_token_embeddings(len(self.bert_tokenizer))
            self.Qformer.cls = None
            self.t5_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
            )
            self.ln_vision = LayerNorm(self.visual_encoder.num_features)
            print("Loading QFormer Done")

        if self.visual_input != "QFORMER":
            print("Loading Vision Aggregator")
            self.ln_adapter = LayerNorm(self.visual_encoder.num_features)
            self.adapter_proj = nn.Sequential(
                nn.Linear(self.visual_encoder.num_features, self.visual_encoder.num_features * 4),
                nn.GELU(),
                nn.Linear(self.visual_encoder.num_features * 4, self.t5_model.config.hidden_size),
            )
            self.fusion_adapter = FusionAdapter(num_blocks=2,dim=self.visual_encoder.num_features)
            print("Loading Vision Aggregator Done")

        if self.enable_semantic_tags:
            tag_sp_token = "<extra_id_0>"
            self.tag_softPrompt_id = self.t5_tokenizer.convert_tokens_to_ids(tag_sp_token)
            self.tag_prompt = "According to <extra_id_0>, you are allowed to use or partially use the following tags: [{}]. "
            self.soft_prompt_hint = nn.Parameter(torch.zeros(self.t5_model.config.hidden_size))
            self.soft_prompt_hint.data.normal_(mean=0.0, std=self.t5_model.config.initializer_factor)
        logging.info(f"boost_lr_scale:{boost_lr_scale}")
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def _init_vision_encoder(
        self, model_path, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        print("Using normal vit")
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision, model_path
        )
        return visual_encoder
    
    def _init_Qformer(self, bert_model, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(bert_model)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def _init_llm(self, llm_model):
        print("Loading LLM")
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(llm_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(llm_model, truncation_side='right')

        llm_config = T5Config.from_pretrained(llm_model)
        llm_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            llm_model, config=llm_config, torch_dtype=torch.bfloat16,
        )
        set_adapter_t5(self.t5_model, self.t5_model.config.d_model, n=2 if self.visual_input=="ALL" else 1, bottleneck=64)
        
        for name, param in self.t5_model.named_parameters():
            if "adapter" in name:
                if "router_ratio" in name and self.visual_input != "ALL":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        print("Loading LLM Done")
    
    def _init_ram(self):
        if self.ram_model == None:
            print("Loading RAM Model For Tag Generation")
            self.ram_model = ram(pretrained=self.ram_path, image_size=384, vit="swin_l", text_encoder_type=self.bert_model).cuda()
            self.ram_processor = get_transform()
            print("Loading RAM Model Done")
    
    def generate_tags(self, images:Union[List[Image], Image]) -> List[str]:
        """
        Generate tags for provided images.
        
        Args:
            images (Image or List[Image])
        Returns:
            tags (List[str])
        """
        
        self._init_ram()
        if isinstance(images, Image):
            images = [images]
        images = torch.stack([self.ram_processor(img) for img in images]).to(self.device)
        tags = self.ram_model.generate_tag(images, threshold=0.85)[0]
        return [t.replace(" |",",") for t in tags]
    
    def _insert_tags(self, samples, prompt):
        if self.enable_semantic_tags:
            assert self.tag_prompt is not None, "Please provide Tags prompt."
            if "tags" not in samples:
                samples = self._generate_tags(samples)
            prompt = [self.tag_prompt.format(tags) + tin for tags, tin in zip(samples["tags"], prompt)]
        return prompt
    
    def _insert_softTagHint(self, samples, input_tokens, inputs_embeds):
        if self.enable_semantic_tags:
            bs = inputs_embeds.size(0)
            sp_embeds = self.soft_prompt_hint.expand(bs, -1).to(inputs_embeds.dtype)
            sp_index = (input_tokens.input_ids == self.tag_softPrompt_id).nonzero(as_tuple=True)
            inputs_embeds[sp_index] = sp_embeds
        return inputs_embeds
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        p_boost, p_boost_non_wd = [], []
        boost_name = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if "fusion_adapter" in n:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_boost_non_wd.append(p)
                else:
                    p_boost.append(p)
                boost_name.append(n)
            else:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
            {"params": p_boost, "weight_decay": weight_decay, "lr_scale": lr_scale*self.boost_lr_scale},
            {"params": p_boost_non_wd, "weight_decay": 0, "lr_scale": lr_scale*self.boost_lr_scale},
        ]
        logging.info(f"boost params:{boost_name}")
        return optim_params

    def encode_img(self, image, question):
        with self.maybe_autocast(dtype=torch.bfloat16):
            image_embeds, intermediate = self.visual_encoder(image, return_intermediate=True)
            if self.visual_input != "QFORMER":
                adapter_embeds = self.ln_adapter(self.fusion_adapter(intermediate[38], [intermediate[28],intermediate[18]]))
                adapter_embeds = self.adapter_proj(adapter_embeds)
            if self.visual_input != "AGGREGATOR":
                image_embeds = self.ln_vision(image_embeds)
        if self.visual_input != "AGGREGATOR":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text_Qformer = self.bert_tokenizer(
                question,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            input_qformer = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        match self.visual_input:
            case "AGGREGATOR":
                inputs_t5 = adapter_embeds
            case "QFORMER":
                inputs_t5 = input_qformer
            case "ALL":
                inputs_t5 = torch.cat([input_qformer, adapter_embeds],dim=1)
            case _:
                raise NotImplementedError(f"Visual input type {self.visual_input} is not supported.")
                
        atts_t5 = torch.ones(inputs_t5.size()[:-1],dtype=torch.long).to(image.device)
    
        return inputs_t5, atts_t5

    def forward(self, samples):
        img_embeds, img_atts = self.encode_img(samples["image"], samples["question"])
        prompt = self._insert_tags(samples, samples["question"])
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            output_tokens = self.t5_output_tokenizer(
                samples["answer"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(self.device)
            
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            text_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            text_embeds = self._insert_softTagHint(samples, input_tokens, text_embeds)
            text_atts = input_tokens.attention_mask

            input_embeds = torch.cat([img_embeds, text_embeds], dim=1)
            encoder_atts = torch.cat([img_atts, text_atts], dim=1)

            set_router_idx(self.t5_model, int(samples["category"][0] != "region_level"))
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        img_embeds, img_atts = self.encode_img(samples["image"].to(self.device), samples["question"])
        prompt = self._insert_tags(samples, samples["question"])
        input_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            text_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            text_embeds = self._insert_softTagHint(samples, input_tokens, text_embeds)
            text_atts = input_tokens.attention_mask
            
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
            input_atts = torch.cat([img_atts, text_atts], dim=1)
            set_router_idx(self.t5_model, int(samples.get("category") != "region_level"))
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=input_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    @classmethod
    def from_config(cls, cfg):
        bert_model = cfg.get("bert_model")
        vit_model = cfg.get("vit_model")
        llm_model = cfg.get("llm_model")
        ram_model = cfg.get("ram_model")
        
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 128)
        visual_input = cfg.get("visual_input", "ALL")
        enable_semantic_tags = cfg.get("enable_semantic_tags", True)
        boost_lr_scale = cfg.get("boost_lr_scale", 1.0)
        
        img_size = cfg.get("image_size", 224)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        num_query_token = cfg.get("num_query_token", 32)
        
        model = cls(
            bert_model=bert_model,
            vit_model=vit_model,
            llm_model=llm_model,
            ram_model=ram_model,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            visual_input=visual_input,
            enable_semantic_tags=enable_semantic_tags,
            boost_lr_scale=boost_lr_scale,
            
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
        )

        model.load_checkpoint_from_config(cfg)

        return model
