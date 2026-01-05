from transformers import AutoConfig
from typing import Optional


def ensure_attn_implementation_in_config(config: Optional[object], attn_implementation: Optional[str]) -> None:
    """
    Ensure that _attn_implementation is set in the config if both config and attn_implementation are provided.
    
    This is necessary because when a config is passed explicitly to from_pretrained(),
    transformers may not automatically set _attn_implementation in the config object.
    
    Args:
        config: The model config object (can be None)
        attn_implementation: The attention implementation string (e.g., "sdpa", "flash_attention_2", "eager")
    """
    if config is not None and attn_implementation is not None:
        setattr(config, "_attn_implementation", attn_implementation)


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
