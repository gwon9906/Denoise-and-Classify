import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_cae_multitask(input_shape_img=(32, 32, 3), num_classes=10) -> Model:
    """Multitask convolutional autoencoder WITHOUT skip connections (replacement).

    - Restoration head: decoder upsamples from bottleneck only (no same-scale concat)
    - Classification head: global average pooled bottleneck → dense classifier
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # Encoder (4 layers)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    p1 = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    p2 = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    p3 = layers.MaxPooling2D(2)(x)  # 4x4

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    p4 = layers.MaxPooling2D(2)(x)  # 2x2

    # Bottleneck (1 layer)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # Classification head from bottleneck
    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.5)(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', name='classification_output')(feat)

    # Decoder (4 layers)
    d = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # 4x4
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d)  # 8x8
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d)  # 16x16
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)  # 32x32
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    rec = layers.Conv2D(3, 1, activation='linear', name='restoration_output')(d)
    return Model(inputs=img_in, outputs=[rec, cls_out], name='CAE_multitask')


def build_cae_restoration(input_shape_img=(32, 32, 3)) -> Model:
    """Restoration-only autoencoder WITHOUT skip connections (replacement)."""
    img_in = layers.Input(shape=input_shape_img)

    # Encoder (4 layers)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    p1 = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    p2 = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    p3 = layers.MaxPooling2D(2)(x)  # 4x4

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    p4 = layers.MaxPooling2D(2)(x)  # 2x2

    # Bottleneck (1 layer)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # Decoder (4 layers)
    d = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # 4x4
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d)  # 8x8
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d)  # 16x16
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)  # 32x32
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    rec = layers.Conv2D(3, 1, activation='linear')(d)
    return Model(inputs=img_in, outputs=rec, name='CAE_restoration')


# Removed deprecated no-skip aliases to avoid duplication; use build_cae_multitask/build_cae_restoration above.
