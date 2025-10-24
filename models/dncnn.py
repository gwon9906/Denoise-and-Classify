from tensorflow.keras import layers, Model, regularizers
import tensorflow as tf


def conv_block_dncnn(x, filters, use_bn=True, activation='relu'):
    """DnCNN convolution block with optional batch normalization."""
    x = layers.Conv2D(filters, 3, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x


def build_dncnn_multitask(input_shape_img=(32, 32, 3), num_classes=10, 
                         depth=17, filters=64, use_bn=True) -> Model:
    """Modern DnCNN-style multitask model with improved architecture.

    Args:
        input_shape_img: Input image shape
        num_classes: Number of classification classes
        depth: Number of convolution layers (typically 17 or 20)
        filters: Number of filters in each layer
        use_bn: Whether to use batch normalization

    Returns:
        Compiled DnCNN multitask model
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU (with bias)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(img_in)
    
    # Hidden layers: Conv + BN + ReLU (no bias)
    for i in range(depth - 2):
        x = conv_block_dncnn(x, filters, use_bn=use_bn, activation='relu')
    
    # Last layer: Conv (no activation, no bias)
    res = layers.Conv2D(3, 3, padding='same', activation='linear', 
                       use_bias=False, name='residual_pred')(x)
    
    # Restoration output: input - predicted noise
    rec = layers.Subtract(name='restoration_output')([img_in, res])

    # Classification head from bottleneck features
    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(256, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.5)(feat)
    feat = layers.Dense(128, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.3)(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(feat)

    return Model(inputs=img_in, outputs=[rec, cls_out], 
                name=f'DnCNN_multitask_{depth}L_{filters}F')


def build_dncnn_restoration(input_shape_img=(32, 32, 3), depth=17, 
                           filters=64, use_bn=True) -> Model:
    """Modern DnCNN restoration-only model with improved architecture.

    Args:
        input_shape_img: Input image shape
        depth: Number of convolution layers
        filters: Number of filters in each layer
        use_bn: Whether to use batch normalization

    Returns:
        DnCNN restoration model
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU (with bias)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(img_in)
    
    # Hidden layers: Conv + BN + ReLU (no bias)
    for i in range(depth - 2):
        x = conv_block_dncnn(x, filters, use_bn=use_bn, activation='relu')
    
    # Last layer: Conv (no activation, no bias)
    res = layers.Conv2D(3, 3, padding='same', activation='linear', 
                       use_bias=False)(x)
    
    # Restoration output: input - predicted noise
    rec = layers.Subtract(name='restoration_output')([img_in, res])
    
    return Model(inputs=img_in, outputs=rec, 
                name=f'DnCNN_restoration_{depth}L_{filters}F')


def build_dncnn_modern(input_shape_img=(32, 32, 3), num_classes=10, 
                      depth=20, filters=64, use_bn=True, 
                      residual_learning=True) -> Model:
    """Modern DnCNN with additional improvements.

    Args:
        input_shape_img: Input image shape
        num_classes: Number of classification classes
        depth: Number of convolution layers
        filters: Number of filters in each layer
        use_bn: Whether to use batch normalization
        residual_learning: Whether to use residual learning (predict noise)

    Returns:
        Modern DnCNN model
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(img_in)
    
    # Hidden layers with skip connections every 5 layers
    skip_connections = []
    for i in range(depth - 2):
        x = conv_block_dncnn(x, filters, use_bn=use_bn, activation='relu')
        
        # Add skip connection every 5 layers
        if (i + 1) % 5 == 0 and i < depth - 3:
            skip_connections.append(x)
    
    # Last layer: Conv (no activation, no bias)
    if residual_learning:
        # Predict noise residual
        res = layers.Conv2D(3, 3, padding='same', activation='linear', 
                           use_bias=False, name='residual_pred')(x)
        rec = layers.Subtract(name='restoration_output')([img_in, res])
    else:
        # Direct image prediction
        rec = layers.Conv2D(3, 3, padding='same', activation='linear', 
                           name='restoration_output')(x)

    # Classification head
    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(512, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.5)(feat)
    feat = layers.Dense(256, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(feat)
    feat = layers.Dropout(0.3)(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(feat)

    return Model(inputs=img_in, outputs=[rec, cls_out], 
                name=f'DnCNN_modern_{depth}L_{filters}F')


# Backward compatibility aliases
build_dncnn_baseline = build_dncnn_multitask
