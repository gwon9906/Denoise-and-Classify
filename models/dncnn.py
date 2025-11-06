from tensorflow.keras import layers, Model, regularizers
import tensorflow as tf


def conv_block_dncnn(input_tensor, num_filters, use_batch_norm=True, activation='relu'):
    """DnCNN convolution block with optional batch normalization."""
    output_tensor = layers.Conv2D(num_filters, 3, padding='same', use_bias=not use_batch_norm)(input_tensor)
    if use_batch_norm:
        output_tensor = layers.BatchNormalization()(output_tensor)
    if activation:
        output_tensor = layers.Activation(activation)(output_tensor)
    return output_tensor


def build_dncnn_multitask(input_shape_img=(32, 32, 3), num_classes=10, 
                         depth=17, num_filters=64, use_batch_norm=True) -> Model:
    """Modern DnCNN-style multitask model with improved architecture.

    Args:
        input_shape_img: Input image shape
        num_classes: Number of classification classes
        depth: Number of convolution layers (typically 17 or 20)
        num_filters: Number of filters in each layer
        use_batch_norm: Whether to use batch normalization

    Returns:
        Compiled DnCNN multitask model
    """
    image_input = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU (with bias)
    network_layer = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(image_input)
    
    # Hidden layers: Conv + BN + ReLU (no bias)
    for layer_idx in range(depth - 2):
        network_layer = conv_block_dncnn(network_layer, num_filters, use_batch_norm=use_batch_norm, activation='relu')
    
    # Last layer: Conv (no activation, no bias)
    residual_prediction = layers.Conv2D(3, 3, padding='same', activation='linear', 
                       use_bias=False, name='residual_pred')(network_layer)
    
    # Restoration output: input - predicted noise
    restoration_output = layers.Subtract(name='restoration_output')([image_input, residual_prediction])

    # Classification head from bottleneck features
    features = layers.GlobalAveragePooling2D()(network_layer)
    features = layers.Dense(256, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(features)
    features = layers.Dropout(0.5)(features)
    features = layers.Dense(128, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(features)
    features = layers.Dropout(0.3)(features)
    classification_output = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(features)

    return Model(inputs=image_input, outputs=[restoration_output, classification_output], 
                name=f'DnCNN_multitask_{depth}L_{num_filters}F')


def build_dncnn_restoration(input_shape_img=(32, 32, 3), depth=17, 
                           num_filters=64, use_batch_norm=True) -> Model:
    """Modern DnCNN restoration-only model with improved architecture.

    Args:
        input_shape_img: Input image shape
        depth: Number of convolution layers
        num_filters: Number of filters in each layer
        use_batch_norm: Whether to use batch normalization

    Returns:
        DnCNN restoration model
    """
    image_input = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU (with bias)
    network_layer = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(image_input)
    
    # Hidden layers: Conv + BN + ReLU (no bias)
    for layer_idx in range(depth - 2):
        network_layer = conv_block_dncnn(network_layer, num_filters, use_batch_norm=use_batch_norm, activation='relu')
    
    # Last layer: Conv (no activation, no bias)
    residual_prediction = layers.Conv2D(3, 3, padding='same', activation='linear', 
                       use_bias=False)(network_layer)
    
    # Restoration output: input - predicted noise
    restoration_output = layers.Subtract(name='restoration_output')([image_input, residual_prediction])
    
    return Model(inputs=image_input, outputs=restoration_output, 
                name=f'DnCNN_restoration_{depth}L_{num_filters}F')


def build_dncnn_modern(input_shape_img=(32, 32, 3), num_classes=10, 
                      depth=20, num_filters=64, use_batch_norm=True, 
                      residual_learning=True) -> Model:
    """Modern DnCNN with additional improvements.

    Args:
        input_shape_img: Input image shape
        num_classes: Number of classification classes
        depth: Number of convolution layers
        num_filters: Number of filters in each layer
        use_batch_norm: Whether to use batch normalization
        residual_learning: Whether to use residual learning (predict noise)

    Returns:
        Modern DnCNN model
    """
    image_input = layers.Input(shape=input_shape_img, name='image_input')

    # First layer: Conv + ReLU
    network_layer = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(image_input)
    
    # Hidden layers with skip connections every 5 layers
    skip_connections = []
    for layer_idx in range(depth - 2):
        network_layer = conv_block_dncnn(network_layer, num_filters, use_batch_norm=use_batch_norm, activation='relu')
        
        # Add skip connection every 5 layers
        if (layer_idx + 1) % 5 == 0 and layer_idx < depth - 3:
            skip_connections.append(network_layer)
    
    # Last layer: Conv (no activation, no bias)
    if residual_learning:
        # Predict noise residual
        residual_prediction = layers.Conv2D(3, 3, padding='same', activation='linear', 
                           use_bias=False, name='residual_pred')(network_layer)
        restoration_output = layers.Subtract(name='restoration_output')([image_input, residual_prediction])
    else:
        # Direct image prediction
        restoration_output = layers.Conv2D(3, 3, padding='same', activation='linear', 
                           name='restoration_output')(network_layer)

    # Classification head
    features = layers.GlobalAveragePooling2D()(network_layer)
    features = layers.Dense(512, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(features)
    features = layers.Dropout(0.5)(features)
    features = layers.Dense(256, activation='relu', 
                       kernel_regularizer=regularizers.l2(1e-4))(features)
    features = layers.Dropout(0.3)(features)
    classification_output = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(features)

    return Model(inputs=image_input, outputs=[restoration_output, classification_output], 
                name=f'DnCNN_modern_{depth}L_{num_filters}F')


# Backward compatibility aliases
build_dncnn_baseline = build_dncnn_multitask
