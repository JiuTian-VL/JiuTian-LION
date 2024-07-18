import os
import torch
import torch.nn as nn
import logging
from omegaconf import OmegaConf

class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_pretrained = cfg.get("load_pretrained", True)
        if load_pretrained:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert pretrain_path, "pretrained path is not specified in the config file"
            print("loading pretrain: ", pretrain_path)
            msg = self.load_checkpoint(filename=pretrain_path, **kwargs)
            print("unexpected_keys: ", msg.unexpected_keys)
    
    def load_checkpoint(self, filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info(f"Unexpected keys: {msg.unexpected_keys}")
        logging.info(f"load checkpoint from: {filename}")

        return msg

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]
    
    def counting_training_parameters(self):
        total = 0.
        trainable_names = []
        all = 0.
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.nelement()
                trainable_names.append(name)
            all += param.nelement()
        logging.info(trainable_names)
        logging.info('  + Number of trainable params: %.2fM' % (total / 1e6))
        logging.info('Number of all params: %.2fM' % (all / 1e6))
        return total
                
    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError()
                
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model
                

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)        
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
        ]                
        return optim_params
    