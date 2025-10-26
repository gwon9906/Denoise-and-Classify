import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TiedDense(layers.Layer):
    """
    BAM의 양방향 연상 원리를 구현한 Tied Dense 레이어.
    decoder에서 encoder Dense의 kernel을 전치해서 재사용하여 양방향 연상 학습을 구현.
    """
    def __init__(self, tied_to: layers.Dense, activation=None, use_bias=True, name=None):
        super().__init__(name=name)
        self.tied_to = tied_to
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):   
        if self.use_bias:
            # tied_to.kernel shape: (input_dim, output_dim)
            # transpose 후: (output_dim, input_dim)
            # 따라서 출력 차원은 tied_to의 input_dim
            tied_output_dim = self.tied_to.kernel.shape[0]
            self.bias = self.add_weight(
                shape=(tied_output_dim,), 
                initializer="zeros", 
                trainable=True, 
                name="bias"
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        kernel_T = tf.transpose(self.tied_to.kernel)
        out = tf.linalg.matmul(inputs, kernel_T)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        """직렬화를 위한 설정 반환"""
        config = super().get_config()
        config.update({
            "activation": keras.activations.serialize(self.activation) if self.activation else None,
            "use_bias": self.use_bias,
        })
        return config


class BAMLayer(layers.Layer):
    """
    BAM의 핵심: 양방향 연상 메모리 레이어
    입력 패턴과 출력 패턴 간의 연관을 양방향으로 학습
    """
    def __init__(self, units, activation='tanh', name=None):
        super().__init__(name=name)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='bam_weights'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Forward pass: 입력 -> 출력
        forward = tf.matmul(inputs, self.W)
        forward = self.activation(forward)
        
        # Backward pass: 출력 -> 입력 (양방향 연상)
        backward = tf.matmul(forward, tf.transpose(self.W))
        
        return forward, backward
    
    def get_config(self):
        """직렬화를 위한 설정 반환"""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation_name,
        })
        return config
