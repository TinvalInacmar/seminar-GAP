
_base_ = [
    '../_base_/default_runtime.py',
    f'../_base_/datasets/FERPlus.py',
]
vit_small = dict(img_size=112, patch_size=16, embed_dim=768, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6)
model = dict(
    type='PoolingVitClassifier',
    extractor=dict(
        type='IRSE',
        input_size=(112, 112),
        pretrained='weights/backbone_weights.pth',
        num_layers=50,
        mode='ir',
        return_index=[2],   # only use the first 3 stages
        return_type='Tensor',
        multi_finetune=True
    ),
    convert=None,
    vit=dict(
        type='PoolingViT',
        pretrained='./weights//vit_small_p16_224-15ec54c9.pth',#../../../input/vit-small/vit_small_p16_224-15ec54c9.pth',
        input_type='feature',
        patch_num=196,
        in_channels=[256],
        attn_method='SUM_ABS_1',
        sum_batch_mean=False,
        cnn_pool_config=dict(keep_num=160, exclude_first=False),
        vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 4, exclude_first=True, attn_method='SUM'),  # None by default
        depth=8,
        **vit_small,
    ),
    head=dict(
        type='CFTreeHead',
        pretrained='weights/head_weights.pth',
        neg_label_in_coarse=0,
        neg_num_classes=6, #
        coarse_num_classes=3,
        in_channels=256,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ))
)


data = dict(
    samples_per_gpu=128,    # total batch size: 128
    workers_per_gpu=8,
)


optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)

optimizer_config = dict(grad_clip=dict(max_norm=10.0, norm_type=2))


lr_config = dict(
    policy='CosineAnnealing', min_lr=0.,
)
log_config = dict(interval=20)

# find_unused_parameters = True

#distributed = True

