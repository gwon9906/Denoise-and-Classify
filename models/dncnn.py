from tensorflow.keras import layers, Model, regularizers


def build_dncnn_multitask(input_shape_img=(32, 32, 3), num_classes=10, depth=17, filters=64) -> Model:
    """DnCNN-style multitask model (no conditioning).

    - Restoration head: residual learning (predict noise, subtract)
    - Classification head: global average pooled features
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(img_in)
    for _ in range(depth - 2):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    res = layers.Conv2D(3, 3, padding='same', activation='linear', name='residual_pred')(x)
    rec = layers.Subtract(name='restoration_output')([img_in, res])

    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.5)(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', name='classification_output')(feat)

    return Model(inputs=img_in, outputs=[rec, cls_out], name='DnCNN_multitask_basic')


def build_dncnn_restoration(input_shape_img=(32, 32, 3), depth=17, filters=64) -> Model:
    """DnCNN restoration-only model (residual denoising)."""
    img_in = layers.Input(shape=input_shape_img)

    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(img_in)
    for _ in range(depth - 2):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    res = layers.Conv2D(3, 3, padding='same', activation='linear')(x)
    rec = layers.Subtract()([img_in, res])
    return Model(inputs=img_in, outputs=rec, name='DnCNN_restoration_basic')
