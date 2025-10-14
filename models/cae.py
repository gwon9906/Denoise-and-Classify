import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_cae_multitask(input_shape_img=(32, 32, 3), num_classes=10) -> Model:
    """Basic autoencoder-style multitask model (no conditioning).

    - Restoration head: U-Net style upsampling with same-scale skip connections
    - Classification head: global average pooled bottleneck → dense classifier
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    skip_1 = x  # 32x32
    p1 = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip_2 = x  # 16x16
    p2 = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    # Classification head from bottleneck
    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.5)(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', name='classification_output')(feat)

    # Decoder (restoration head)
    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)  # 16x16
    d = layers.Concatenate()([d, skip_2])
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)  # 32x32
    d = layers.Concatenate()([d, skip_1])
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    rec = layers.Conv2D(3, 1, activation='linear', name='restoration_output')(d)
    return Model(inputs=img_in, outputs=[rec, cls_out], name='CAE_multitask_basic')


def build_cae_restoration(input_shape_img=(32, 32, 3)) -> Model:
    """Basic restoration-only autoencoder (no conditioning)."""
    img_in = layers.Input(shape=input_shape_img)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    skip_1 = x
    p1 = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip_2 = x
    p2 = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    d = layers.Concatenate()([d, skip_2])
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)
    d = layers.Concatenate()([d, skip_1])
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    rec = layers.Conv2D(3, 1, activation='linear')(d)
    return Model(inputs=img_in, outputs=rec, name='CAE_restoration_basic')
