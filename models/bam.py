# BAM (Bidirectional Associative Memory) 개선 버전
import tensorflow as tf
from tensorflow.keras import layers
from .bam_mf import soft_sign

# ============================================
# BAM 반복 안정화 함수 (soft sign 기반)
# ============================================

def bam_associative_recall(x, y, w_xy, w_yx, steps=5, temp=1.0):
    """
    BAM 양방향 연상 반복
    
    Args:
        x: 입력 패턴 (batch, input_dim)
        y: 출력 패턴 초기값 (batch, output_dim)
        w_xy: X->Y 가중치 (input_dim, output_dim)
        w_yx: Y->X 가중치 (output_dim, input_dim)
        steps: 반복 횟수
        temp: temperature parameter
        
    Returns:
        안정화된 출력 패턴 y
    """
    for _ in range(steps):
        # X -> Y 방향
        y = soft_sign(tf.matmul(x, w_xy), temp=temp)
        # Y -> X 방향 (안정화)
        x = soft_sign(tf.matmul(y, w_yx), temp=temp)
    
    return y


# ============================================
# BAM 연상 계층 (soft sign 반복 안정화)
# ============================================

class BAMAssociativeLayer(layers.Layer):
    """
    BAM Associative Layer with iterative recall
    
    특징:
    - 양방향 가중치 (Wxy, Wyx)
    - Soft sign 기반 반복 안정화
    - Temperature 조절 가능
    """
    
    def __init__(self, input_dim, output_dim, steps=5, temp=1.0, name=None):
        super(BAMAssociativeLayer, self).__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steps = steps
        self.temp = temp
    
    def build(self, input_shape):
        # X -> Y 가중치
        self.Wxy = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wxy'
        )
        
        # Y -> X 가중치 (feedback)
        self.Wyx = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wyx'
        )
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        # 초기 출력 패턴 (0으로 시작)
        y0 = tf.zeros((tf.shape(x)[0], self.output_dim))
        
        # 반복 연상으로 안정화
        y = bam_associative_recall(
            x, y0, self.Wxy, self.Wyx,
            steps=self.steps,
            temp=self.temp
        )
        
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'steps': self.steps,
            'temp': self.temp
        })
        return config


# ============================================
# 기존 BAM 클래스 (Hebbian 학습용, 호환성 유지)
# ============================================

class BAM(tf.keras.Model):
    """
    BAM 분류기 (Hebbian 학습 지원)
    
    기존 코드와의 호환성을 위해 유지
    """
    
    def __init__(self, input_dim, output_dim, delta=0.2):
        """
        Args:
            input_dim: BAM 입력 벡터 차원
            output_dim: 출력 벡터 차원 (클래스 개수)
            delta: 활성화 함수 파라미터 (사용 안함, 호환성용)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.delta = delta
        
        # 가중치 행렬 W (input_dim x output_dim)
        self.W = tf.Variable(
            tf.zeros([input_dim, output_dim]),
            trainable=True,
            name='W'
        )
    
    def call(self, x):
        """정방향 연상: 입력 x에 대한 출력 패턴 y 반환"""
        y_lin = tf.matmul(x, self.W)
        # Soft sign 사용 (더 안정적)
        y = soft_sign(y_lin, temp=1.0)
        return y
    
    def train_hebbian(self, X, Y, epochs=1, eta=0.008):
        """
        Hebbian 학습 규칙에 따른 가중치 훈련
        
        Args:
            X: 학습 입력 배열
            Y: 대응 학습 출력(정답) 배열
            epochs: 전체 패턴 반복 학습 횟수
            eta: 학습률
        """
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        n = tf.shape(X)[0]
        
        for ep in range(epochs):
            # 패턴 순서를 섞어 학습
            indices = tf.random.shuffle(tf.range(n))
            for i in indices:
                x0 = tf.expand_dims(X[i], 0)  # (1, input_dim)
                y0 = tf.expand_dims(Y[i], 0)  # (1, output_dim)
                
                # 정방향으로 출력 패턴 계산
                y_pred = self(x0)
                
                # 역방향으로 입력 패턴 회상
                x_pred = soft_sign(
                    tf.matmul(y0, self.W, transpose_b=True),
                    temp=1.0
                )
                
                # Hebbian 가중치 변화 계산
                dW = tf.matmul(
                    (x0 + x_pred),
                    (y0 - y_pred),
                    transpose_a=True
                )
                
                # 가중치 업데이트
                self.W.assign_add(eta * dW)


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Testing BAM modules...")
    
    # BAMAssociativeLayer 테스트
    import numpy as np
    x_dummy = np.random.randn(10, 256).astype('float32')
    
    bam_layer = BAMAssociativeLayer(
        input_dim=256,
        output_dim=10,
        steps=5,
        temp=0.5
    )
    
    y_pred = bam_layer(x_dummy)
    
    print(f"Input shape: {x_dummy.shape}")
    print(f"Output shape: {y_pred.shape}")
    print("✓ BAMAssociativeLayer test passed")
    
    # BAM 클래스 테스트
    bam = BAM(input_dim=256, output_dim=10)
    y_pred2 = bam(x_dummy)
    
    print(f"BAM output shape: {y_pred2.shape}")
    print("✓ BAM class test passed")
