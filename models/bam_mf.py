# MF-BAM 개선 버전 (극저 SNR 대응, 드롭아웃 제거)
import tensorflow as tf
from tensorflow.keras import layers

# ============================================
# 포화성 큐빅 활성화 함수
# ============================================

def cubic_activation(input_tensor):
    """
    단순 큐빅 활성화: x - x^3/3 (안정화 버전)
    입력을 clipping하여 수치 안정성 보장
    """
    # ✅ 입력 clipping으로 수치 안정성 확보 (NaN 방지)
    clipped_input = tf.clip_by_value(input_tensor, -10.0, 10.0)
    return clipped_input - tf.pow(clipped_input, 3) / 3.0


# Soft Sign 함수 (BAM 안정화용)


def soft_sign(input_tensor, temp=1.0):
    """
    소프트 사인 함수 (tanh 기반)
    temp: temperature parameter (낮을수록 sharp)
    """
    return tf.tanh(input_tensor / temp)


# HebbianDenseLayer (W, V 분리, BatchNorm 사용)


class HebbianDenseLayer(layers.Layer):
    """
    Hebbian Dense Layer with feedback connections
    
    특징:
    - W (feedforward), V (feedback) 가중치 분리
    - BatchNormalization 사용 (GPU 호환)
    - Cubic activation
    """
    
    def __init__(self, units, activation=None, use_feedback=True, name=None):
        super(HebbianDenseLayer, self).__init__(name=name)
        self.units = units
        self.activation = activation if activation is not None else cubic_activation
        self.use_feedback = use_feedback
    
    def build(self, input_shape):
        input_dimension = input_shape[-1]
        
        # Feedforward 가중치 W
        self.feedforward_weights = self.add_weight(
            shape=(input_dimension, self.units),
            initializer='glorot_uniform',  # ✅ glorot이 cubic activation에 더 안정적
            trainable=True,
            name='feedforward_weights'
        )
        
        # Feedback 가중치 V (선택적, 현재 미사용)
        # if self.use_feedback:
        #     self.feedback_weights = self.add_weight(
        #         shape=(self.units, input_dimension),
        #         initializer='he_uniform',
        #         trainable=True,
        #         name='feedback_weights'
        #     )
        
        # ✅ BatchNormalization 안정화 설정
        self.normalization_layer = layers.BatchNormalization(
            momentum=0.99,  # 더 안정적인 이동 평균
            epsilon=1e-3,    # 수치 안정성
            center=True,
            scale=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Feedforward: x -> W -> z
        layer_output = tf.matmul(inputs, self.feedforward_weights)
        layer_output = self.normalization_layer(layer_output, training=training)
        
        # Activation
        if self.activation is not None:
            layer_output = self.activation(layer_output)
        
        return layer_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_feedback': self.use_feedback
        })
        return config


# ============================================
# MF 모듈 (개선 버전)
# ============================================

class MF(tf.keras.Model):
    """
    Multifactor (MF) Autoencoder
    
    개선 사항:
    - HebbianDenseLayer 사용
    - Dropout 제거 (극저 SNR 대응)
    - BatchNormalization 사용 (GPU 호환)
    - Cubic activation
    """
    
    def __init__(self, input_dim, encoder_dims=[1024, 768, 512, 256]):
        """
        Args:
            input_dim: 입력 차원 (CIFAR-10: 3072)
            encoder_dims: 인코더 은닉층 차원 목록
        """
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        
        # 인코더 (특성 추출) 층 생성
        self.encoder_layers = []
        for i, dim in enumerate(encoder_dims):
            layer = HebbianDenseLayer(
                dim,
                activation=cubic_activation,
                use_feedback=True,
                name=f"mf_encoder_{i+1}"
            )
            self.encoder_layers.append(layer)
        
        # 디코더 (복원) 층 생성
        decoder_dims = encoder_dims[-2::-1]  # 역순 (마지막 제외)
        decoder_dims.append(input_dim)
        
        self.decoder_layers = []
        for i, dim in enumerate(decoder_dims):
            # 마지막 레이어는 sigmoid activation
            if i == len(decoder_dims) - 1:
                layer = layers.Dense(
                    dim,
                    activation='sigmoid',
                    name='restoration_output'
                )
            else:
                layer = HebbianDenseLayer(
                    dim,
                    activation=cubic_activation,
                    use_feedback=True,
                    name=f"mf_decoder_{i+1}"
                )
            self.decoder_layers.append(layer)
    
    def encode(self, inputs, training=None):
        """입력 x를 잠재 표현 z로 인코딩"""
        encoded = inputs
        for layer in self.encoder_layers:
            encoded = layer(encoded, training=training)
        return encoded
    
    def decode(self, latent_representation, training=None):
        """잠재 표현 z로부터 입력 형태로 복원"""
        decoded = latent_representation
        for layer in self.decoder_layers:
            decoded = layer(decoded, training=training)
        return decoded
    
    def call(self, inputs, training=None):
        """
        MF 모듈을 통해 입력을 복원하고 잠재 표현도 반환
        
        Returns:
            (복원 이미지, 잠재 표현)
        """
        # ✅ 입력 검증 및 clipping (NaN 방지)
        inputs = tf.clip_by_value(inputs, 0.0, 1.0)
        
        latent_representation = self.encode(inputs, training=training)
        
        # ✅ 잠재 표현 안정화 (gradient explosion 방지)
        latent_representation = tf.clip_by_value(latent_representation, -100.0, 100.0)
        
        restored_output = self.decode(latent_representation, training=training)
        
        # ✅ 출력 안정화 (sigmoid 후에도 보장)
        restored_output = tf.clip_by_value(restored_output, 0.0, 1.0)
        
        return restored_output, latent_representation
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'encoder_dims': self.encoder_dims
        }


# ============================================
# 편의 함수: MF 모듈 빌드
# ============================================

def build_mf_module(input_vector, hidden_units=[1024, 768, 512, 256]):
    """
    함수형 API로 MF 모듈 빌드
    
    Args:
        input_vector: 입력 텐서
        hidden_units: 은닉층 차원 리스트
        
    Returns:
        잠재 표현 텐서
    """
    current_layer = input_vector
    for layer_idx, units in enumerate(hidden_units):
        current_layer = HebbianDenseLayer(
            units,
            activation=cubic_activation,
            name=f"mf_layer_{layer_idx+1}"
        )(current_layer)
    return current_layer


def build_decoder(latent_representation, decoder_units=[512, 768, 1024], original_dimension=3072):
    """
    함수형 API로 디코더 빌드
    
    Args:
        latent_representation: 잠재 표현 텐서
        decoder_units: 디코더 은닉층 차원 리스트
        original_dimension: 원본 입력 차원
        
    Returns:
        복원된 출력 텐서
    """
    current_layer = latent_representation
    for layer_idx, units in enumerate(decoder_units):
        current_layer = HebbianDenseLayer(
            units,
            activation=cubic_activation,
            name=f"decoder_layer_{layer_idx+1}"
        )(current_layer)
    
    return layers.Dense(
        original_dimension,
        activation='sigmoid',
        name='restoration_output'
    )(current_layer)


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Testing MF module...")
    
    # MF 모듈 생성
    mf = MF(input_dim=3072, encoder_dims=[1024, 768, 512, 256])
    
    # 더미 데이터
    import numpy as np
    x_dummy = np.random.randn(10, 3072).astype('float32')
    
    # Forward pass
    restored, latent = mf(x_dummy)
    
    print(f"Input shape: {x_dummy.shape}")
    print(f"Restored shape: {restored.shape}")
    print(f"Latent shape: {latent.shape}")
    print("✓ MF module test passed")
