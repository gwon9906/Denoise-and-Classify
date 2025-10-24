# BAM (Bidirectional Associative Memory) 복원 전용 모델
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TiedDense(layers.Layer):
    """
    BAM의 양방향 연상 원리를 구현한 Tied Dense 레이어.
    decoder에서 encoder Dense의 kernel을 전치해서 재사용하여 양방향 연상 학습을 구현.
    """
    def __init__(self, tied_to: layers.Dense, units: int, activation=None, use_bias=True, name=None):
        super().__init__(name=name)
        self.tied_to = tied_to
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="bias"
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        # encoder kernel shape: (in_dim, out_dim)
        # decoder wants: inputs @ (out_dim, in_dim)  => output shape: (batch, in_dim)
        kernel_T = tf.transpose(self.tied_to.kernel)
        out = tf.linalg.matmul(inputs, kernel_T)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out


class BAMLayer(layers.Layer):
    """
    BAM의 핵심: 양방향 연상 메모리 레이어 (복원 전용)
    노이즈 패턴과 깨끗한 패턴 간의 연관을 양방향으로 학습
    """
    def __init__(self, units, activation='tanh', name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # BAM의 가중치 행렬 (양방향 연상을 위한)
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='bam_weights'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Forward pass: 노이즈 입력 -> 깨끗한 특징
        forward = tf.matmul(inputs, self.W)
        forward = self.activation(forward)
        
        # Backward pass: 깨끗한 특징 -> 복원된 입력 (양방향 연상)
        backward = tf.matmul(forward, tf.transpose(self.W))
        
        return forward, backward


def build_bam_restoration_model(input_dim, latent_dim=128):
    """
    BAM 기반 복원 전용 모델 구축
    
    Args:
        input_dim: 입력 차원 (예: 32*32*3 = 3072)
        latent_dim: 잠재 공간 차원
    
    Returns:
        BAM 기반 복원 모델
    """
    # 입력
    inp = keras.Input(shape=(input_dim,), name="noisy_input")

    # --- Encoder (Forward Pass) ---
    enc1 = layers.Dense(1024, activation="relu", name="enc_dense1")
    h1 = enc1(inp)

    enc2 = layers.Dense(256, activation="relu", name="enc_dense2")
    h2 = enc2(h1)

    # BAM Layer: 양방향 연상 메모리 (노이즈 -> 깨끗한 패턴)
    bam_layer = BAMLayer(latent_dim, activation='tanh', name="bam_memory")
    z, z_backward = bam_layer(h2)

    # --- Decoder (Backward Pass with Tied Weights) ---
    # BAM의 양방향 연상 원리: tied weights로 역방향 복원
    d1 = TiedDense(tied_to=enc2, units=256, activation="relu", name="dec_tied1")(z)
    d2 = TiedDense(tied_to=enc1, units=1024, activation="relu", name="dec_tied2")(d1)
    recon_logits = TiedDense(tied_to=enc1, units=input_dim, activation=None, name="dec_tied3")(d2)
    recon = layers.Activation("sigmoid", name="restored_img")(recon_logits)

    model = keras.Model(inputs=inp, outputs=recon, name="BAM_Restoration")
    return model


def create_bam_restoration_model(input_dim=3072, latent_dim=128, learning_rate=1e-3):
    """
    BAM 복원 모델 생성 및 컴파일
    
    Args:
        input_dim: 입력 차원 (32*32*3 = 3072)
        latent_dim: 잠재 공간 차원
        learning_rate: 학습률
    
    Returns:
        컴파일된 BAM 복원 모델
    """
    model = build_bam_restoration_model(input_dim, latent_dim)
    
    # 손실 함수 (복원 전용)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse"), 
                keras.metrics.MeanAbsoluteError(name="mae")]
    )
    
    return model


def preprocess_cifar10_restoration_data(noise_std=0.2, seed=42):
    """
    CIFAR-10 복원용 데이터 전처리
    
    Args:
        noise_std: 노이즈 표준편차
        seed: 랜덤 시드
    
    Returns:
        전처리된 데이터 딕셔너리
    """
    # CIFAR-10 데이터 로드
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # [0,1] 정규화 & 평탄화
    def preprocess(x):
        x = x.astype("float32") / 255.0
        return x.reshape((x.shape[0], -1))

    x_train_clean = preprocess(x_train)
    x_test_clean = preprocess(x_test)

    # 노이즈 버전 만들기
    rng = np.random.default_rng(seed)
    x_train_noisy = x_train_clean + rng.normal(0.0, noise_std, x_train_clean.shape).astype("float32")
    x_test_noisy = x_test_clean + rng.normal(0.0, noise_std, x_test_clean.shape).astype("float32")
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    return {
        'x_train_clean': x_train_clean,
        'x_test_clean': x_test_clean,
        'x_train_noisy': x_train_noisy,
        'x_test_noisy': x_test_noisy
    }


if __name__ == "__main__":
    # 모델 생성 예제
    model = create_bam_restoration_model(input_dim=3072, latent_dim=128)
    model.summary()
    
    # 데이터 전처리 예제
    data = preprocess_cifar10_restoration_data(noise_std=0.2, seed=42)
    print(f"Training data shape: {data['x_train_noisy'].shape}")
    print(f"Test data shape: {data['x_test_noisy'].shape}")
