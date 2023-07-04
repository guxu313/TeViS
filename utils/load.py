import torch
from src.utils.logger import LOGGER


def load_model_weights_with_mismatch(model, loaded_path, load_swin=False, load_bert=False, pretrained2d=False, change_window=False):
    """operated in-place, no need to return `model`"""

    if load_swin:
        loaded_state_dict = process_swin_weights(model, loaded_path, pretrained2d)
    elif load_bert:
        loaded_state_dict = torch.load(loaded_path, map_location="cpu")
    else:
        loaded_state_dict = torch.load(loaded_path, map_location="cpu")

    if change_window and not load_bert and not load_swin:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in loaded_state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del loaded_state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in loaded_state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del loaded_state_dict[k]

        # delete relative_position_bias_table since we always re-init it
        relative_position_bias_table_keys = [k for k in loaded_state_dict.keys() if "relative_position_bias_table" in k]

        # bicubic interpolate relative_position_bias_table if not match
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = loaded_state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]

            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()

            i_layer = int(k.split('.blocks')[0].split('.')[-1])

            layer_window_size = model.video_encoder.window_size[i_layer]

            L2 = (2*layer_window_size[0]-1) * (2*layer_window_size[1]-1) * (2*layer_window_size[2]-1)

            if nH1 != nH2:
                LOGGER.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    W1 = (2*layer_window_size[1]-1) * (2*layer_window_size[2]-1)
 
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, -1, W1).float(), size=(2*layer_window_size[0]-1, W1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                    loaded_state_dict[k] = relative_position_bias_table_pretrained.half()
     
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model and loaded, but shape mismatched:")
    LOGGER.info(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    model.load_state_dict(toload, strict=False)

    del loaded_state_dict, toload, mismatched_shape_keys


