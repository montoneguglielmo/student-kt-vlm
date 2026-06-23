import torch
import torch.nn as nn
 
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model


# --------------------------------------------------------------------------- #
# Model: FROZEN backbone + trainable linear head                              #
# --------------------------------------------------------------------------- #


class KTProbe(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )

        # Freeze EVERYTHING in the backbone.
        for p in self.vlm.parameters():
            p.requires_grad = False
        self.vlm.eval()                            # disable dropout etc. in backbone

        # The only trainable part.
        hidden = self.vlm.config.text_config.hidden_size
        self.head = nn.Linear(hidden, 1, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        # No grad through the backbone -> saves memory and time.
        with torch.no_grad():
            out = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs,
            )
            last_hidden = out.hidden_states[-1]        # (B, T, hidden)

        # Hidden state of the last non-padding token in each sequence.
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lengths]   # (B, hidden)

        logits = self.head(pooled).squeeze(-1)         # (B,)
        return logits
    
# --------------------------------------------------------------------------- #
# Model: LoRA-adapted backbone + trainable linear head                        #
# --------------------------------------------------------------------------- #
class KTLoRA(nn.Module):
    def __init__(self, model_id, r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=None):
        super().__init__()

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )
        hidden = vlm.config.text_config.hidden_size

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.vlm = get_peft_model(vlm, lora_config)
        self.vlm.print_trainable_parameters()

        self.head = nn.Linear(hidden, 1, dtype=torch.bfloat16)


    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        # Grad flows through the LoRA adapters now -> no torch.no_grad().
        out = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden = out.hidden_states[-1]            # (B, T, hidden)

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lengths]   # (B, hidden)

        logits = self.head(pooled).squeeze(-1)         # (B,)
        return logits
    
    
# --------------------------------------------------------------------------- #
# Build the model specified in the cfg file                                   #
# --------------------------------------------------------------------------- #
def build_model(cfg):
    """Build a model from the config based on cfg.model_type."""
    if cfg.model_type == "probe":
        return KTProbe(cfg.model_id)
    elif cfg.model_type == "lora":
        return KTLoRA(
            cfg.model_id,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
        )