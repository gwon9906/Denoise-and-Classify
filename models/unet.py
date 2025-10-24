import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters: int):
    """Standard UNet convolution block: Conv → BN → ReLU → Conv → BN → ReLU"""
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet_multitask(input_shape_img=(32, 32, 3), num_classes=10, 
                        base_filters=64, depth=5, residual=True):
    """UNet with both restoration and classification heads.
    
    Args:
        input_shape_img: Input image shape
        num_classes: Number of classification classes
        base_filters: Number of filters in the first layer
        depth: Number of encoder/decoder levels
        residual: If True, predict noise residual and subtract from input
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
    
    # Restoration output
    if residual:
        res = layers.Conv2D(3, 3, padding='same', activation='linear', name='residual_pred')(x)
        rec = layers.Subtract(name='restoration_output')([img_in, res])
    else:
        rec = layers.Conv2D(3, 1, padding='same', activation='linear', name='restoration_output')(x)
    
    return Model(inputs=img_in, outputs=[rec, cls_out], 
                name=f'UNet_multitask_{base_filters}x{depth}_{"res" if residual else "direct"}')

def build_unet_restoration(input_shape=(32, 32, 3), base_filters=64, depth=4, residual=True):
    """UNet model for restoration only (no classification head).
    
    Args:
        input_shape: Input image shape
        base_filters: Number of filters in the first layer
        depth: Number of encoder/decoder levels
        residual: If True, predict noise residual and subtract from input
    """
    assert depth >= 2, "UNet depth must be >= 2"
    
    img_in = layers.Input(shape=input_shape, name='image_input')
    
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
    
    # Decoder
    for level in reversed(range(depth - 1)):
        filters //= 2
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block(x, filters)
    
    # Restoration output
    if residual:
        res = layers.Conv2D(3, 3, padding='same', activation='linear', name='residual_pred')(x)
        rec = layers.Subtract(name='restoration_output')([img_in, res])
    else:
        rec = layers.Conv2D(3, 1, padding='same', activation='linear', name='restoration_output')(x)
    
    return Model(inputs=img_in, outputs=rec, 
                name=f'UNet_restoration_{base_filters}x{depth}_{"res" if residual else "direct"}')

# Backward compatibility aliases
build_unet_baseline = build_unet_multitask