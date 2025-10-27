from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import torch


def modality_ids_to_modality_attention_mask(modality_ids, input_shape, torch_dtype):
    assert modality_ids.ndim == 2
    assert modality_ids.unique().max() <= 1 # text: 0, image: 1
    assert modality_ids.unique().min() >= 0 # text: 0, image: 1

    # input_shape is (batch_size, seq_length)
    _, s = input_shape
    query_length = modality_ids.shape[1]

    # mask_img is (batch_size, 1, query_length, s)
    mask_img = AttentionMaskConverter(is_causal=False).to_4d(modality_ids, query_length, torch_dtype)
    mask_img = combine_attention_masks(mask_img, mask_img.transpose(2, 3))

    # mask_text is (batch_size, 1, query_length, s)
    inverted_modality_ids = 1 - modality_ids
    mask_text = AttentionMaskConverter(is_causal=False).to_4d(inverted_modality_ids, query_length, torch_dtype)
    mask_text = combine_attention_masks(mask_text, mask_text.transpose(2, 3))

    mask_combined = mask_img.masked_fill(mask_text == 0., 0.) # OR operation

    # return the mask with shape (batch_size, 1, s, s) with mask values set to -inf to isolate the text and image tokens
    return mask_combined[:, :, :s, :]

def combine_attention_masks(mask, *masks):
    # combine the mask using AND operation
    combined_mask = mask
    for m in masks:
        assert m.shape == combined_mask.shape
        assert m.dtype == combined_mask.dtype

        combined_mask = combined_mask.masked_fill(m != 0., torch.finfo(combined_mask.dtype).min)

    return combined_mask
