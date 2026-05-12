# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _build_grad_scaler(mixed_precision=False):
    if not mixed_precision:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def split_checkpoint_weights(state_dict):
    encoder, predictor = {}, {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_k = k
            if "embeddings.patch_embeddings." in k:
                new_k = new_k.replace("embeddings.patch_embeddings.", "patch_embed.")
            if ".layer." in k:  # 更严格的匹配
                new_k = new_k.replace(".layer.", ".blocks.")
            new_k = new_k.replace("encoder.", "module.", 1)
            encoder[new_k] = v
        elif "predictor" in k.lower() or k.startswith("predictor"):
            predictor[k.replace("predictor.", "module.")] = v

    return encoder, predictor


def load_pretrained_safetensors(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    encoder_dict, predictor_dict = split_checkpoint_weights(checkpoint)
    if load_encoder:
        # -- loading encoder
        missing, unexpected = encoder.load_state_dict(encoder_dict, strict=False)
        if len(missing) != 0:
            logger.info(f"loaded pretrained encoder missing: {missing}")
        if len(unexpected) != 0:
            logger.info(f"loaded pretrained encoder unexpected: {unexpected}")

    if load_predictor:
        # -- loading predictor
        missing, unexpected = predictor.load_state_dict(predictor_dict, strict=False)
        # if len(missing) != 0:
        # logger.info(f"loaded pretrained encoder missing: {missing}")
        # if len(unexpected) != 0:
        # logger.info(f"loaded pretrained encoder unexpected: {unexpected}")

    # -- loading target_encoder
    if load_encoder:
        missing, unexpected = target_encoder.load_state_dict(encoder_dict, strict=False)
        if len(missing) != 0:
            logger.info(f"loaded pretrained encoder missing: {missing}")
        if len(unexpected) != 0:
            logger.info(f"loaded pretrained encoder unexpected: {unexpected}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=None,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    # epoch = checkpoint["epoch"]
    breakpoint()
    if load_encoder:
        # -- loading encoder
        breakpoint()
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        # logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        # logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if load_encoder:
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint[target_encoder_key]
            pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
            msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
            # logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )


def load_checkpoint(
    r_path,
    encoder,
    predictor=None,
    target_encoder=None,
    student_perceiver=None,
    teacher_perceiver=None,
    opt=None,
    scaler=None,
    seg_head=None,
    replace_kw=[],
):
    breakpoint()
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if predictor is not None:
        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    if student_perceiver is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["student_perceiver"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = student_perceiver.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained student_perceiver from epoch {epoch} with msg: {msg}")

    if teacher_perceiver is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["teacher_perceiver"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = teacher_perceiver.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained teacher_perceiver from epoch {epoch} with msg: {msg}")

    if seg_head is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["seg_head"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = seg_head.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained teacher_perceiver from epoch {epoch} with msg: {msg}")
    # -- loading optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        # predictor,
        seg_head,
        # target_encoder,
        student_perceiver,
        # teacher_perceiver,
        opt,
        scaler,
        epoch,
    )


def load_checkpoint_resample_neck_head(
    r_path,
    encoder,
    seg_neck=None,
    student_perceiver=None,
    opt=None,
    scaler=None,
    seg_head=None,
    replace_kw=[],
):
    # breakpoint()
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if student_perceiver is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["student_perceiver"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = student_perceiver.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained student_perceiver from epoch {epoch} with msg: {msg}")

    if seg_head is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["seg_head"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = seg_head.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained seg_head from epoch {epoch} with msg: {msg}")

    if seg_neck is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["seg_neck"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = seg_neck.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained seg_neck from epoch {epoch} with msg: {msg}")
    # -- loading optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        # predictor,
        seg_head,
        seg_neck,
        # target_encoder,
        # student_perceiver,
        # teacher_perceiver,
        opt,
        scaler,
        epoch,
    )


def load_checkpoint_neck_head(
    r_path,
    encoder,
    seg_neck=None,
    student_perceiver=None,
    opt=None,
    scaler=None,
    seg_head=None,
    replace_kw=[],
):
    # breakpoint()
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if seg_head is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["seg_head"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = seg_head.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained seg_head from epoch {epoch} with msg: {msg}")

    if seg_neck is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["seg_neck"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = seg_neck.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained seg_neck from epoch {epoch} with msg: {msg}")
    # -- loading optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        # predictor,
        seg_head,
        seg_neck,
        # target_encoder,
        # student_perceiver,
        # teacher_perceiver,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    use_extrinsics=False,
    old_pred=False,
    use_perceiver_ema=False,
    target_shape=None,
):
    # encoder = video_vit.__dict__[model_name](
    #     img_size=crop_size,
    #     patch_size=patch_size,
    #     num_frames=max_num_frames,
    #     tubelet_size=tubelet_size,
    #     uniform_power=uniform_power,
    #     use_sdpa=use_sdpa,
    #     use_silu=use_silu,
    #     wide_silu=wide_silu,
    #     use_activation_checkpointing=use_activation_checkpointing,
    #     use_rope=use_rope,
    # )

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=1024,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
        use_perceiver_ema=use_perceiver_ema,
        target_shape=target_shape,
    )

    # encoder.to(device)
    predictor.to(device)
    # logger.info(encoder)
    logger.info(predictor)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    # logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    # return encoder,predictor
    return predictor


def init_predictor_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    embed_dim=1024,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    state_embed_dim=None,
    use_extrinsics=False,
    command_dim=0,
    old_pred=False,
    use_perceiver_ema=False,
    target_shape=None,
):

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        state_embed_dim=state_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
        use_perceiver_ema=use_perceiver_ema,
        target_shape=target_shape,
        command_dim=command_dim,
    )
    predictor.to(device)
    logger.info(predictor)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    # logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return predictor


def init_opt_no_resample_world_model(
    encoder,
    predictor,
    seg_head,
    seg_neck,
    # student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = []

    if predictor is not None:
        param_groups.extend(
            [
                {
                    "params": (
                        p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
                    ),
                },
                {
                    "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                    "WD_exclude": zero_init_bias_wd,
                    "weight_decay": 0,
                },
            ]
        )

    # 添加 seg_neck 参数 (如果存在)
    if seg_neck is not None:
        param_groups.extend(
            [
                {
                    "params": (p for n, p in seg_neck.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
                    "lr_scale": enc_lr_scale,
                },
                {
                    "params": (p for n, p in seg_neck.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                    "WD_exclude": zero_init_bias_wd,
                    "weight_decay": 0,
                    "lr_scale": enc_lr_scale,
                },
            ]
        )

    # 添加 seg_head 参数 (如果存在)
    if seg_head is not None:
        param_groups.extend(
            [
                {
                    "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
                },
                {
                    "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                    "WD_exclude": zero_init_bias_wd,
                    "weight_decay": 0,
                },
            ]
        )

    # 分类统计
    predictor_params = (
        sum(p.numel() for n, p in predictor.named_parameters() if p.requires_grad) if predictor is not None else 0
    )
    seg_neck_params = (
        sum(p.numel() for n, p in seg_neck.named_parameters() if p.requires_grad) if seg_neck is not None else 0
    )
    seg_head_params = (
        sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad) if seg_head is not None else 0
    )

    total_trainable = predictor_params + seg_head_params + seg_neck_params

    # logger.info("=" * 80)
    # logger.info("🔍 Trainable Parameters Breakdown:")
    # if seg_neck is not None:
    #     logger.info(f"  seg_neck:        {seg_neck_params / 1e6:>8.2f}M")
    # logger.info(f"  predictor_params:     {predictor_params / 1e6:>8.2f}M")
    # if seg_head is not None:
    #     logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    # logger.info(f"  {'─' * 40}")
    # logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    # logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_resample_world_model(
    encoder,
    predictor,
    seg_head,
    seg_neck,
    student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (
                p for n, p in student_perceiver.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in student_perceiver.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # 分类统计
    seg_neck_params = sum(p.numel() for n, p in seg_neck.named_parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for n, p in predictor.named_parameters() if p.requires_grad)
    student_perceiver_params = sum(p.numel() for n, p in student_perceiver.named_parameters() if p.requires_grad)
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    if seg_head_params != 0:
        total_trainable = predictor_params + student_perceiver_params + seg_head_params + seg_neck_params
        # if encoder_params != 0:
        #     total_trainable =  predictor_params + student_perceiver_params + seg_head_params + encoder_params
    else:
        total_trainable = predictor_params + student_perceiver_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  seg_neck:        {seg_neck_params / 1e6:>8.2f}M")
    logger.info(f"  predictor_params:     {predictor_params / 1e6:>8.2f}M")
    logger.info(f"  student_perceiver_params: {student_perceiver_params / 1e6:>8.2f}M")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt(
    encoder,
    predictor,
    seg_head,
    student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (
                p for n, p in student_perceiver.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in student_perceiver.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # 分类统计
    encoder_params = sum(p.numel() for n, p in encoder.named_parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for n, p in predictor.named_parameters() if p.requires_grad)
    student_perceiver_params = sum(p.numel() for n, p in student_perceiver.named_parameters() if p.requires_grad)
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    if seg_head_params != 0:
        total_trainable = predictor_params + student_perceiver_params + seg_head_params + encoder_params
        # if encoder_params != 0:
        #     total_trainable =  predictor_params + student_perceiver_params + seg_head_params + encoder_params
    else:
        total_trainable = predictor_params + student_perceiver_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  encoder:        {encoder_params / 1e6:>8.2f}M")
    logger.info(f"  predictor_params:     {predictor_params / 1e6:>8.2f}M")
    logger.info(f"  student_perceiver_params: {student_perceiver_params / 1e6:>8.2f}M")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_only_seg(
    seg_head,
    student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (
                p for n, p in student_perceiver.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in student_perceiver.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # 分类统计
    student_perceiver_params = sum(p.numel() for n, p in student_perceiver.named_parameters() if p.requires_grad)
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    total_trainable = student_perceiver_params + seg_head_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  student_perceiver_params: {student_perceiver_params / 1e6:>8.2f}M")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


import logging

import torch

logger = logging.getLogger(__name__)


def init_opt_new(
    model_list,  # 变更点：传入一个列表或字典，包含所有模型组件
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    enc_lr_scale=1.0,
):
    """
    通用优化器初始化函数。
    自动检测 requires_grad，自动分离 Weight Decay。
    """

    # 1. 定义参数组容器
    # group_regular: 需要 Weight Decay (通常是 weights)
    # group_no_decay: 不需要 Weight Decay (通常是 bias, layernorm, embedding)
    # group_encoder: 如果 encoder 需要单独的 lr_scale，可以单独处理

    param_groups = []

    # 辅助函数：判断是否应该应用 Weight Decay
    def needs_wd(name, param):
        # 常见的不做 WD 的层
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embedding" in name:
            return False
        return True

    total_trainable_params = 0

    # 2. 遍历传入的所有模型模块
    # model_list 结构示例: [{"model": encoder, "lr_scale": 0.1}, {"model": predictor, "lr_scale": 1.0}]
    # 或者简单列表: [encoder, predictor, seg_head] (默认 scale 1.0)

    logger.info("=" * 80)
    logger.info("🔍 Initializing Optimizer - Dynamic Parameter Grouping")

    for item in model_list:
        if isinstance(item, dict):
            module = item["model"]
            scale = item.get("lr_scale", 1.0)
            name_prefix = item.get("name", "unknown_module")
        else:
            module = item
            scale = 1.0
            name_prefix = module.__class__.__name__

        if module is None:
            continue

        # 统计该模块参数
        module_trainable = 0
        decay_params = []
        no_decay_params = []

        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue

            module_trainable += p.numel()

            if needs_wd(n, p):
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        total_trainable_params += module_trainable

        if module_trainable > 0:
            logger.info(
                f"  Add to Opt: {name_prefix:<20} | Trainable: {module_trainable/1e6:>6.2f}M | LR Scale: {scale}"
            )

            # 添加到参数组
            if decay_params:
                param_groups.append(
                    {
                        "params": decay_params,
                        "weight_decay": wd,  # 初始 WD，会被 scheduler 覆盖
                        "lr_scale": scale,
                    }
                )

            if no_decay_params:
                param_groups.append(
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "lr_scale": scale,
                    }
                )

    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL TRAINABLE PARAMS: {total_trainable_params / 1e6:>8.2f}M")
    logger.info("=" * 80)

    # 3. 初始化优化器
    optimizer = torch.optim.AdamW(param_groups, lr=ref_lr, betas=betas, eps=eps)

    # 4. 初始化 Scheduler (保持你原有的逻辑不变)
    # 注意：WSDSchedule 和 CosineWDSchedule 需要能够处理 param_groups 里的 "lr_scale" 字段
    # 如果你的 WSDSchedule 没有处理 lr_scale，请确保它读取的是 group['lr'] * group.get('lr_scale', 1.0)

    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )

    scaler = _build_grad_scaler(mixed_precision)

    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_seg_neck(
    seg_head,
    seg_neck,
    student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (
                p for n, p in student_perceiver.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in student_perceiver.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # 分类统计
    student_perceiver_params = sum(p.numel() for n, p in student_perceiver.named_parameters() if p.requires_grad)
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    seg_neck_params = sum(p.numel() for n, p in seg_neck.named_parameters() if p.requires_grad)
    total_trainable = student_perceiver_params + seg_head_params + seg_neck_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  student_perceiver_params: {student_perceiver_params / 1e6:>8.2f}M")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  seg_neck_params: {seg_neck_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_seg_neck_no_resample(
    seg_head,
    seg_neck,
    student_perceiver,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    encoder=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in seg_neck.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    if encoder is not None:
        param_groups = [
            {
                "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            },
            {
                "params": (p for n, p in seg_neck.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            },
            {
                "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
            {
                "params": (p for n, p in seg_neck.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
                "lr_scale": enc_lr_scale,
            },
            {
                "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
                "lr_scale": enc_lr_scale,
            },
        ]
    # 分类统计
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    seg_neck_params = sum(p.numel() for n, p in seg_neck.named_parameters() if p.requires_grad)
    total_trainable = seg_head_params + seg_neck_params
    if encoder is not None:
        encoder_params = sum(p.numel() for n, p in encoder.named_parameters() if p.requires_grad)
        total_trainable += encoder_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  seg_neck_params: {seg_neck_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_noresample(
    encoder,
    seg_head,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": 0.5 * enc_lr_scale,
        },
        {
            "params": (p for n, p in seg_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # 分类统计
    seg_head_params = sum(p.numel() for n, p in seg_head.named_parameters() if p.requires_grad)
    total_trainable = seg_head_params
    logger.info("=" * 80)
    logger.info("🔍 Trainable Parameters Breakdown:")
    logger.info(f"  seg_head_params: {seg_head_params / 1e6:>8.2f}M")
    logger.info(f"  {'─' * 40}")
    logger.info(f"  TOTAL:               {total_trainable / 1e6:>8.2f}M")
    logger.info("=" * 80)
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = _build_grad_scaler(mixed_precision)
    return optimizer, scaler, scheduler, wd_scheduler


def resume_training_checkpoint(
    resume_path,
    encoder=None,
    target_encoder=None,
    predictor=None,
    seg_head=None,
    seg_neck=None,
    planner=None,
    optimizer=None,
    scaler=None,
    scheduler=None,
    wd_scheduler=None,
    logger=None,
    rank=0,
    world_size=1,
    use_broadcast=True,
    model_only=False,
):
    """
    从检查点恢复训练状态

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank，
    避免所有 rank 同时从 NFS 读取同一个大文件。

    Args:
        resume_path: 检查点文件路径
        encoder, target_encoder, predictor, seg_head, seg_neck, planner: 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        logger: 日志记录器
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        use_broadcast: 是否使用 broadcast 分发 checkpoint（False 则每个 rank 独立从磁盘读取）
        model_only: 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）

    Returns:
        start_epoch: 恢复的起始epoch（model_only=True 时始终返回 0）
    """
    import torch
    import torch.distributed as dist

    if logger is None:
        logger = logging.getLogger()

    logger.info(f"正在从 {resume_path} 恢复训练...")

    # 多卡场景：仅 rank 0 从磁盘读取，broadcast 给其他 rank
    if use_broadcast and world_size > 1 and dist.is_available() and dist.is_initialized():
        if rank == 0:
            logger.info(f"Rank 0 loading resume checkpoint from disk: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            broadcast_list = [checkpoint]
        else:
            broadcast_list = [None]
        logger.info(f"Rank {rank} waiting for resume checkpoint broadcast...")
        dist.broadcast_object_list(broadcast_list, src=0)
        checkpoint = broadcast_list[0]
        logger.info(f"Rank {rank} received resume checkpoint via broadcast")
    else:
        logger.info(f"Rank {rank} loading resume checkpoint from disk: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

    def summarize_keys(keys, max_items=10):
        keys = list(keys)
        preview = keys[:max_items]
        suffix = "" if len(keys) <= max_items else f", ... (+{len(keys) - max_items} more)"
        return f"{preview}{suffix}"

    def load_state_dict(model, state_dict, name):
        """安全加载状态字典，处理 shape 不匹配的参数和 PEFT 模型"""
        if state_dict is None:
            return

        # 检测 PEFT (LoRA) 包装: DDP(PeftModel(base)) 或 PeftModel(base)
        model_unwrapped = model.module if hasattr(model, "module") else model
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        is_peft = hasattr(model_unwrapped, "base_model") and hasattr(model_unwrapped.base_model, "model")

        if is_peft:
            # PEFT 模型：剥离 module. 前缀后加载到底层 base_model.model
            base_model = model_unwrapped.base_model.model
            # PEFT/LoRA 将 target module 的 weight/bias 移到 base_layer 下,
            # 例如 qkv.weight → qkv.base_layer.weight, 需要自动 remap
            model_keys = set(base_model.state_dict().keys())
            remapped = {}
            for k, v in new_state_dict.items():
                if k in model_keys:
                    remapped[k] = v
                else:
                    # 尝试插入 .base_layer
                    for suffix in (".weight", ".bias"):
                        if k.endswith(suffix):
                            new_key = k[: -len(suffix)] + ".base_layer" + suffix
                            if new_key in model_keys:
                                remapped[new_key] = v
                                break
                    else:
                        remapped[k] = v
            missing, unexpected = base_model.load_state_dict(remapped, strict=False)
            # missing 中应只剩 LoRA adapter 参数 (lora_A/lora_B), 这是正常的
            lora_missing = [k for k in missing if "lora_" in k]
            non_lora_missing = [k for k in missing if "lora_" not in k]
            logger.info(
                f"已恢复 {name} 权重 (PEFT base model): "
                f"missing={len(missing)} (lora={len(lora_missing)}, other={len(non_lora_missing)}), "
                f"unexpected={len(unexpected)}"
            )
            if non_lora_missing:
                logger.warning(f"{name} PEFT base model non-LoRA missing keys: {non_lora_missing[:10]}")
            if unexpected:
                raise RuntimeError(
                    f"{name} PEFT base model has unexpected keys ({len(unexpected)}): {unexpected[:10]}"
                )
            return

        try:
            model_unwrapped.load_state_dict(new_state_dict, strict=True)
            logger.info(f"已恢复 {name} 权重")
        except Exception as strict_error:
            should_raise_mismatch = (not model_only) or name == "planner"
            try:
                missing, unexpected = model_unwrapped.load_state_dict(new_state_dict, strict=False)
            except Exception as relaxed_error:
                message = (
                    f"{name} checkpoint is incompatible with current "
                    f"{model_unwrapped.__class__.__name__}: {relaxed_error}"
                )
                if should_raise_mismatch:
                    raise RuntimeError(message) from relaxed_error
                logger.warning(message)
                return

            if missing or unexpected:
                message = (
                    f"{name} checkpoint is incompatible with current "
                    f"{model_unwrapped.__class__.__name__}: "
                    f"missing={len(missing)} {summarize_keys(missing)}, "
                    f"unexpected={len(unexpected)} {summarize_keys(unexpected)}. "
                    "If this checkpoint is only meant to initialize shared encoder/predictor weights, "
                    "disable loading this module (for example, meta.load_planner=False with resume_model_only=True)."
                )
                if should_raise_mismatch:
                    raise RuntimeError(message) from strict_error
                logger.warning(message)

    # 恢复模型权重
    if encoder is not None and "encoder" in checkpoint:
        load_state_dict(encoder, checkpoint["encoder"], "encoder")
    if target_encoder is not None and "target_encoder" in checkpoint:
        load_state_dict(target_encoder, checkpoint["target_encoder"], "target_encoder")
    if predictor is not None and "predictor" in checkpoint:
        load_state_dict(predictor, checkpoint["predictor"], "predictor")
    if seg_head is not None and "seg_head" in checkpoint:
        load_state_dict(seg_head, checkpoint["seg_head"], "seg_head")
    if seg_neck is not None and "seg_neck" in checkpoint:
        load_state_dict(seg_neck, checkpoint["seg_neck"], "seg_neck")
    if planner is not None and "planner" in checkpoint:
        load_state_dict(planner, checkpoint["planner"], "planner")

    # 恢复 optimizer 状态
    if not model_only and optimizer is not None and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])
        logger.info("已恢复 optimizer 状态")

    # 恢复 scaler 状态
    if not model_only and scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info("已恢复 scaler 状态")

    # 恢复 scheduler 状态
    if not model_only and scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("已恢复 scheduler 状态")

    # 恢复 wd_scheduler 状态
    if not model_only and wd_scheduler is not None and "wd_scheduler" in checkpoint:
        wd_scheduler.load_state_dict(checkpoint["wd_scheduler"])
        logger.info("已恢复 wd_scheduler 状态")

    # 获取 epoch
    if model_only:
        start_epoch = 0
        logger.info("model_only=True: 跳过 optimizer/scheduler/epoch，将从 epoch 0 开始新阶段训练")
    else:
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"已恢复 epoch，将从 epoch {start_epoch} 继续训练")

    logger.info(f"训练已从 {resume_path} 成功恢复！")

    del checkpoint
    return start_epoch
