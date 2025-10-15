import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters: int):
    # Conv → BN → ReLU × 2
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet_baseline(input_shape_img=(32, 32, 3), num_classes=10,
                        base_filters: int = 64, depth: int = 3,
                        residual: bool = True) -> Model:
    """Baseline UNet (single-input, no conditioning) with multitask heads.

    - Encoder grows channels (e.g., 64→128→256 for depth=3), decoder shrinks.
    - residual=True: predict noise residual and subtract from input.
    """
    assert depth >= 2, "UNet depth must be >= 2"

    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # Encoder
    skips = []
    x = img_in
    filters = base_filters
    for level in range(depth):
        x = conv_block(x, filters)
        if level < depth - 1:
            skips.append(x)
            x = layers.MaxPooling2D(2)(x)
            filters *= 2

    bottleneck = x

    # Classification head from bottleneck
    gap = layers.GlobalAveragePooling2D()(bottleneck)
    cls_out = layers.Dense(num_classes, activation='softmax', name='classification_output')(gap)

    # Decoder
    x = bottleneck
    for level in reversed(range(depth - 1)):
        filters //= 2
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block(x, filters)

    if residual:
        res = layers.Conv2D(3, 3, padding='same', activation='linear', name='residual_pred')(x)
        rec = layers.Subtract(name='restoration_output')([img_in, res])
    else:
        rec = layers.Conv2D(3, 1, padding='same', activation='linear', name='restoration_output')(x)

    return Model(inputs=img_in, outputs=[rec, cls_out], name=f'UNet_baseline_{base_filters}x{depth}_{"res" if residual else "direct"}')
def build_unet_multitask(input_shape_img=(32, 32, 3), num_classes=10, residual: bool = True) -> Model:
    """Pure UNet baseline (no conditioning) with multitask heads.

    - If residual=True, restoration head predicts residual and subtracts from input.
      Otherwise, restoration head predicts the clean image directly.
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    c1 = conv_block(img_in, 32)
    p1 = layers.MaxPooling2D(2)(c1)  # ← 풀링 위치는 동일, BN은 conv_block 내부에 반영됨

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)

    # Classification head from bottleneck
    gap = layers.GlobalAveragePooling2D()(c3)
    cls_out = layers.Dense(num_classes, activation='softmax', name='classification_output')(gap)

    # Decoder
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32)

    if residual:
        res = layers.Conv2D(3, 3, padding='same', activation='linear', name='residual_pred')(u1)
        rec = layers.Subtract(name='restoration_output')([img_in, res])
    else:
        rec = layers.Conv2D(3, 1, padding='same', activation='linear', name='restoration_output')(u1)

    return Model(inputs=img_in, outputs=[rec, cls_out], name=f'UNet_multitask_basic_{"res" if residual else "direct"}')