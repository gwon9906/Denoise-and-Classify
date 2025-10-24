# BAM (Bidirectional Associative Memory) 모델 정의
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
    BAM의 핵심: 양방향 연상 메모리 레이어
    입력 패턴과 출력 패턴 간의 연관을 양방향으로 학습
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
        # Forward pass: 입력 -> 출력
        forward = tf.matmul(inputs, self.W)
        forward = self.activation(forward)
        
        # Backward pass: 출력 -> 입력 (양방향 연상)
        backward = tf.matmul(forward, tf.transpose(self.W))
        
        return forward, backward


def build_bam_model(input_dim, latent_dim=128, num_classes=10):
    """
    BAM (Bidirectional Associative Memory) 기반 MTL 모델 구축
    
    Args:
        input_dim: 입력 차원 (예: 32*32*3 = 3072)
        latent_dim: 잠재 공간 차원
        num_classes: 분류할 클래스 수
    
    Returns:
        BAM 기반 MTL 모델
    """
    # 입력
    inp = keras.Input(shape=(input_dim,), name="noisy_input")

    # --- Encoder (Forward Pass) ---
    enc1 = layers.Dense(1024, activation="relu", name="enc_dense1")
    h1 = enc1(inp)

    enc2 = layers.Dense(256, activation="relu", name="enc_dense2")
    h2 = enc2(h1)

    # BAM Layer: 양방향 연상 메모리
    bam_layer = BAMLayer(latent_dim, activation='tanh', name="bam_memory")
    z, z_backward = bam_layer(h2)

    # --- Decoder (Backward Pass with Tied Weights) ---
    # BAM의 양방향 연상 원리: tied weights로 역방향 복원
    d1 = TiedDense(tied_to=enc2, units=256, activation="relu", name="dec_tied1")(z)
    d2 = TiedDense(tied_to=enc1, units=1024, activation="relu", name="dec_tied2")(d1)
    recon_logits = TiedDense(tied_to=enc1, units=input_dim, activation=None, name="dec_tied3")(d2)
    recon = layers.Activation("sigmoid", name="recon_img")(recon_logits)

    # --- Classifier head (BAM의 연상 분류) ---
    # BAM의 연상 메모리를 활용한 분류
    logits = layers.Dense(num_classes, activation=None, name="cls_logits")(z)
    probs = layers.Softmax(name="cls_probs")(logits)

    model = keras.Model(inputs=inp, outputs={"recon": recon, "probs": probs}, name="BAM_MTL")
    return model


def create_bam_model(input_dim=3072, latent_dim=128, num_classes=10, 
                     alpha=1.0, beta=0.5, learning_rate=1e-3):
    """
    BAM 모델 생성 및 컴파일
    
    Args:
        input_dim: 입력 차원 (32*32*3 = 3072)
        latent_dim: 잠재 공간 차원
        num_classes: 분류할 클래스 수
        alpha: reconstruction loss weight
        beta: classification loss weight
        learning_rate: 학습률
    
    Returns:
        컴파일된 BAM 모델
    """
    model = build_bam_model(input_dim, latent_dim, num_classes)
    
    # 손실 함수
    losses = {
        "recon": keras.losses.MeanSquaredError(),
        "probs": keras.losses.CategoricalCrossentropy(from_logits=False)
    }
    loss_weights = {"recon": alpha, "probs": beta}

    # 메트릭
    metrics = {
        "recon": [keras.metrics.MeanSquaredError(name="mse")],
        "probs": [keras.metrics.CategoricalAccuracy(name="acc")]
    }

    # 옵티마이저
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    
    return model


def preprocess_cifar10_data(noise_std=0.2, seed=42):
    """
    CIFAR-10 데이터 전처리 및 노이즈 추가
    
    Args:
        noise_std: 노이즈 표준편차
        seed: 랜덤 시드
    
    Returns:
        전처리된 데이터 딕셔너리
    """
    # CIFAR-10 데이터 로드
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_classes = 10

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

    # 원-핫 인코딩
    y_train_1h = keras.utils.to_categorical(y_train, num_classes)
    y_test_1h = keras.utils.to_categorical(y_test, num_classes)

    return {
        'x_train_clean': x_train_clean,
        'x_test_clean': x_test_clean,
        'x_train_noisy': x_train_noisy,
        'x_test_noisy': x_test_noisy,
        'y_train_1h': y_train_1h,
        'y_test_1h': y_test_1h,
        'y_train': y_train,
        'y_test': y_test,
        'num_classes': num_classes
    }


if __name__ == "__main__":
    # 모델 생성 예제
    model = create_bam_model(input_dim=3072, latent_dim=128, num_classes=10)
    model.summary()
    
    # 데이터 전처리 예제
    data = preprocess_cifar10_data(noise_std=0.2, seed=42)
    print(f"Training data shape: {data['x_train_noisy'].shape}")
    print(f"Test data shape: {data['x_test_noisy'].shape}")
