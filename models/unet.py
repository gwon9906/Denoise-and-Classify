import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x


def build_unet_multitask(input_shape_img=(32, 32, 3), num_classes=10, residual: bool = True) -> Model:
    """Pure UNet baseline (no conditioning) with multitask heads.

    - If residual=True, restoration head predicts residual and subtracts from input.
      Otherwise, restoration head predicts the clean image directly.
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    c1 = conv_block(img_in, 32)
    p1 = layers.MaxPooling2D(2)(c1)

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
