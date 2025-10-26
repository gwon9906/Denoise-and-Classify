# CAE (Convolutional Autoencoder) MTL 모델
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import numpy as np


# ============================================
# 1. MTL CAE 모델
# ============================================

def build_cae_multitask(input_shape_img=(32, 32, 3), num_classes=10,
                       dropout_rate=0.1, l2_reg=1e-4) -> Model:
    """
    Multitask CAE (초저 SNR 최적화, Skip connection 없음)
    
    U-Net과의 비교를 위해 skip connection을 의도적으로 제거.
    순수하게 bottleneck을 통한 정보 압축/복원 능력 평가.
    
    초저 SNR 최적화:
    - 낮은 dropout (0.1 vs 0.5)
    - Sigmoid output activation
    - 대칭적 encoder-decoder 구조
    
    Args:
        input_shape_img: 입력 이미지 shape (32, 32, 3)
        num_classes: 분류 클래스 수
        dropout_rate: Dropout 비율 (초저 SNR: 0.05~0.1)
        l2_reg: L2 regularization 강도
        
    Returns:
        Multi-output model: [restoration_output, classification_output]
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # =====================================
    # Encoder (4 stages)
    # =====================================
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='enc_conv1_1')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='enc_conv1_2')(x)
    p1 = layers.MaxPooling2D(2, name='pool1')(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='enc_conv2_1')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='enc_conv2_2')(x)
    p2 = layers.MaxPooling2D(2, name='pool2')(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='enc_conv3_1')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='enc_conv3_2')(x)
    p3 = layers.MaxPooling2D(2, name='pool3')(x)  # 4x4

    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='enc_conv4_1')(p3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='enc_conv4_2')(x)
    p4 = layers.MaxPooling2D(2, name='pool4')(x)  # 2x2

    # =====================================
    # Bottleneck
    # =====================================
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='bottleneck_conv1')(p4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='bottleneck_conv2')(x)
    bottleneck = x  # 2x2x512

    # =====================================
    # Task 1: Classification Head (from bottleneck)
    # =====================================
    feat = layers.GlobalAveragePooling2D(name='cls_gap')(bottleneck)
    feat = layers.Dense(128, activation='relu', 
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name='cls_dense1')(feat)
    feat = layers.Dropout(dropout_rate, name='cls_dropout')(feat)
    cls_out = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(feat)

    # =====================================
    # Task 2: Decoder (Restoration)
    # NO skip connections - 순수 bottleneck 복원
    # =====================================
    d = layers.Conv2DTranspose(256, 2, strides=2, padding='same', name='dec_upsample1')(bottleneck)  # 4x4
    d = layers.Conv2D(256, 3, padding='same', activation='relu', name='dec_conv1_1')(d)
    d = layers.Conv2D(256, 3, padding='same', activation='relu', name='dec_conv1_2')(d)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding='same', name='dec_upsample2')(d)  # 8x8
    d = layers.Conv2D(128, 3, padding='same', activation='relu', name='dec_conv2_1')(d)
    d = layers.Conv2D(128, 3, padding='same', activation='relu', name='dec_conv2_2')(d)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same', name='dec_upsample3')(d)  # 16x16
    d = layers.Conv2D(64, 3, padding='same', activation='relu', name='dec_conv3_1')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu', name='dec_conv3_2')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same', name='dec_upsample4')(d)  # 32x32
    d = layers.Conv2D(32, 3, padding='same', activation='relu', name='dec_conv4_1')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu', name='dec_conv4_2')(d)

    # Sigmoid activation (픽셀 값 [0,1] 범위 보장)
    rec = layers.Conv2D(3, 1, activation='sigmoid', 
                       name='restoration_output')(d)
    
    return Model(inputs=img_in, outputs=[rec, cls_out], name='CAE_MTL')


def create_cae_mtl(input_shape=(32, 32, 3), num_classes=10,
                   recon_weight=0.7, cls_weight=0.3, learning_rate=1e-3,
                   recon_loss='mae', dropout_rate=0.1):
    """
    CAE MTL 생성 및 컴파일 (one-liner)
    
    Args:
        input_shape: 입력 이미지 shape
        num_classes: 분류할 클래스 수
        recon_weight: reconstruction loss weight (0~1)
        cls_weight: classification loss weight (0~1)
        learning_rate: 학습률
        recon_loss: 'mae', 'mse', 'huber' 중 선택
        dropout_rate: Dropout 비율
        
    Returns:
        컴파일된 CAE MTL 모델
        
    Loss Weight 가이드 (합 = 1):
    - 복원 우선 (초저 SNR): recon_weight=0.7, cls_weight=0.3
    - 분류 우선: recon_weight=0.3, cls_weight=0.7
    - 균형: recon_weight=0.5, cls_weight=0.5
    """
    # Weight 정규화
    total_weight = recon_weight + cls_weight
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
        recon_weight = recon_weight / total_weight
        cls_weight = cls_weight / total_weight
        print(f"Normalized weights - Recon: {recon_weight:.3f}, Cls: {cls_weight:.3f}")
    
    model = build_cae_multitask(input_shape, num_classes, dropout_rate)
    
    # =====================================
    # 손실 함수 선택
    # =====================================
    if recon_loss == 'mae':
        reconstruction_loss = keras.losses.MeanAbsoluteError()
    elif recon_loss == 'mse':
        reconstruction_loss = keras.losses.MeanSquaredError()
    elif recon_loss == 'huber':
        reconstruction_loss = keras.losses.Huber(delta=0.1)
    else:
        raise ValueError(f"Unknown loss: {recon_loss}")
    
    losses = {
        "restoration_output": reconstruction_loss,
        "classification_output": keras.losses.CategoricalCrossentropy()
    }
    
    # Loss weights (합 = 1)
    loss_weights = {
        "restoration_output": recon_weight,
        "classification_output": cls_weight
    }
    
    # 메트릭
    metrics = {
        "restoration_output": [
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ],
        "classification_output": [
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    }
    
    # 옵티마이저
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 모델 컴파일
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model


# ============================================
# 2. CAE MTL 래퍼 클래스
# ============================================

class CAEMTL:
    """
    CAE MTL 래퍼 클래스
    
    복원과 분류를 동시에 학습하는 Convolutional Autoencoder
    - BAM MTL과 비교하기 위한 CNN baseline
    - Skip connection 없음 (U-Net과 구분)
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10,
                 recon_weight=0.7, cls_weight=0.3, learning_rate=1e-3,
                 recon_loss='mae', dropout_rate=0.1):
        """
        Args:
            input_shape: 입력 이미지 shape
            num_classes: 분류할 클래스 수
            recon_weight: reconstruction loss weight (0~1)
            cls_weight: classification loss weight (0~1)
            learning_rate: 학습률
            recon_loss: 'mae', 'mse', 'huber'
            dropout_rate: Dropout 비율
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.recon_loss = recon_loss
        self.dropout_rate = dropout_rate
        
        # Weight 정규화
        total = recon_weight + cls_weight
        self.recon_weight = recon_weight / total
        self.cls_weight = cls_weight / total
        self.learning_rate = learning_rate
        
        # 모델 생성 및 컴파일
        self.model = create_cae_mtl(
            input_shape=input_shape,
            num_classes=num_classes,
            recon_weight=self.recon_weight,
            cls_weight=self.cls_weight,
            learning_rate=learning_rate,
            recon_loss=recon_loss,
            dropout_rate=dropout_rate
        )
    
    def train(self, x_noisy, x_clean, y_labels, epochs=50, batch_size=128,
              validation_split=0.1, callbacks=None):
        """
        CAE MTL 학습
        
        Args:
            x_noisy: 노이즈가 추가된 입력 (N, 32, 32, 3)
            x_clean: 깨끗한 이미지 (복원 타겟)
            y_labels: 원-핫 인코딩된 레이블 (분류 타겟)
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        print("\n" + "="*60)
        print("Training CAE MTL (Multi-Task Learning)")
        print("="*60)
        print(f"Reconstruction loss: {self.recon_loss.upper()}")
        print(f"Loss weights - Restoration: {self.recon_weight:.3f}, Classification: {self.cls_weight:.3f}")
        print(f"Dropout rate: {self.dropout_rate}")
        
        history = self.model.fit(
            x_noisy,
            {
                "restoration_output": x_clean, 
                "classification_output": y_labels
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, x_noisy, batch_size=128, verbose=0):
        """
        CAE MTL 예측
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            batch_size: 배치 크기
            verbose: 진행 표시 레벨
            
        Returns:
            list: [restoration_output, classification_output]
        """
        outputs = self.model.predict(x_noisy, batch_size=batch_size, verbose=verbose)
        
        # outputs는 [restored_images, class_probs] 형태
        return {
            "restoration": outputs[0],
            "classification": outputs[1]
        }
    
    def evaluate(self, x_noisy, x_clean, y_labels, batch_size=128):
        """
        CAE MTL 평가
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            x_clean: 깨끗한 이미지
            y_labels: 원-핫 인코딩된 레이블
            batch_size: 배치 크기
            
        Returns:
            dict: 손실 및 메트릭
        """
        print("\n" + "="*60)
        print("Evaluating CAE MTL")
        print("="*60)
        
        results = self.model.evaluate(
            x_noisy,
            {
                "restoration_output": x_clean,
                "classification_output": y_labels
            },
            batch_size=batch_size,
            verbose=1,
            return_dict=True
        )
        
        # 결과 정리
        eval_results = {
            "total_loss": results['loss'],
            "restoration_loss": results['restoration_output_loss'],
            "classification_loss": results['classification_output_loss'],
            "restoration_mse": results['restoration_output_mse'],
            "restoration_mae": results['restoration_output_mae'],
            "classification_acc": results['classification_output_acc'],
            "top3_acc": results['classification_output_top3_acc']
        }
        
        return eval_results
    
    def evaluate_detailed(self, x_noisy, x_clean, y_labels, batch_size=128):
        """
        상세한 평가 (numpy로 직접 계산)
        
        Returns:
            dict: 상세한 평가 메트릭
        """
        outputs = self.predict(x_noisy, batch_size=batch_size, verbose=1)
        
        # 복원 성능
        restored = outputs['restoration']
        recon_mse = np.mean((restored - x_clean) ** 2)
        recon_mae = np.mean(np.abs(restored - x_clean))
        recon_psnr = 10 * np.log10(1.0 / recon_mse) if recon_mse > 0 else float('inf')
        
        # 분류 성능
        probs = outputs['classification']
        y_pred_classes = np.argmax(probs, axis=1)
        y_true_classes = np.argmax(y_labels, axis=1)
        accuracy = np.mean(y_pred_classes == y_true_classes)
        
        # Top-3 accuracy
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_acc = np.mean([y_true in top3 for y_true, top3 in zip(y_true_classes, top3_preds)])
        
        return {
            "restoration": {
                "mse": float(recon_mse),
                "mae": float(recon_mae),
                "psnr": float(recon_psnr)
            },
            "classification": {
                "accuracy": float(accuracy),
                "top3_accuracy": float(top3_acc)
            }
        }
    
    def save_model(self, filepath="cae_mtl.keras"):
        """
        모델 저장
        
        Args:
            filepath: 저장 경로
        """
        self.model.save(filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath="cae_mtl.keras"):
        """
        모델 로드
        
        Args:
            filepath: 모델 경로
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")
    
    def get_bottleneck_features(self, x_noisy, batch_size=128):
        """
        Bottleneck 표현 추출
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            batch_size: 배치 크기
            
        Returns:
            bottleneck features (N, 2, 2, 512)
        """
        # Bottleneck layer까지의 중간 모델 생성
        bottleneck_layer = None
        for layer in self.model.layers:
            if 'bottleneck' in layer.name:
                bottleneck_layer = layer
                break
        
        if bottleneck_layer is None:
            raise ValueError("Bottleneck layer not found")
        
        # 중간 출력 모델
        intermediate_model = keras.Model(
            inputs=self.model.input,
            outputs=bottleneck_layer.output
        )
        
        features = intermediate_model.predict(x_noisy, batch_size=batch_size)
        return features


# ============================================
# 3. 편의 함수들
# ============================================

def compare_cae_bam_mtl(x_train_noisy, x_train_clean, y_train,
                        x_test_noisy, x_test_clean, y_test,
                        snr_levels=[-30, -20, -10, 0],
                        epochs=30):
    """
    CAE MTL vs BAM MTL 성능 비교
    
    Args:
        데이터셋들
        snr_levels: 테스트할 SNR 레벨들
        epochs: 학습 에폭
        
    Returns:
        비교 결과
    """
    from models.bam_mtl import MTLBAM  # BAM MTL import
    
    results = {
        'CAE_MTL': {},
        'BAM_MTL': {}
    }
    
    for snr in snr_levels:
        print(f"\n{'='*60}")
        print(f"Testing at SNR = {snr} dB")
        print(f"{'='*60}")
        
        # SNR에 맞게 노이즈 추가
        x_noisy_snr = add_noise_by_snr(x_test_clean, snr_db=snr)
        
        # ===== CAE MTL =====
        print("\n--- CAE MTL ---")
        cae = CAEMTL(
            input_shape=(32, 32, 3),
            recon_weight=0.7,
            cls_weight=0.3,
            recon_loss='mae'
        )
        
        cae.train(
            x_train_noisy, x_train_clean, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1
        )
        
        cae_results = cae.evaluate_detailed(x_noisy_snr, x_test_clean, y_test)
        results['CAE_MTL'][snr] = cae_results
        
        # ===== BAM MTL =====
        print("\n--- BAM MTL ---")
        # BAM은 flattened input 필요
        x_train_flat = x_train_noisy.reshape(x_train_noisy.shape[0], -1)
        x_train_clean_flat = x_train_clean.reshape(x_train_clean.shape[0], -1)
        x_test_flat = x_noisy_snr.reshape(x_noisy_snr.shape[0], -1)
        x_test_clean_flat = x_test_clean.reshape(x_test_clean.shape[0], -1)
        
        bam = MTLBAM(
            input_dim=3072,
            recon_weight=0.7,
            cls_weight=0.3,
            recon_loss='mae'
        )
        
        bam.train(
            x_train_flat, x_train_clean_flat, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1
        )
        
        bam_results = bam.evaluate_detailed(x_test_flat, x_test_clean_flat, y_test)
        results['BAM_MTL'][snr] = bam_results
        
        # 결과 출력
        print(f"\nResults at SNR = {snr} dB:")
        print(f"CAE MTL - PSNR: {cae_results['restoration']['psnr']:.2f}, "
              f"Acc: {cae_results['classification']['accuracy']:.4f}")
        print(f"BAM MTL - PSNR: {bam_results['reconstruction']['psnr']:.2f}, "
              f"Acc: {bam_results['classification']['accuracy']:.4f}")
    
    return results


def add_noise_by_snr(clean_data, snr_db):
    """
    특정 SNR로 노이즈 추가
    
    Args:
        clean_data: 깨끗한 데이터
        snr_db: 목표 SNR (dB)
        
    Returns:
        노이즈가 추가된 데이터
    """
    signal_power = np.mean(clean_data ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, clean_data.shape).astype('float32')
    noisy_data = clean_data + noise
    
    return np.clip(noisy_data, 0.0, 1.0)


# ============================================
# 4. 사용 예제
# ============================================

if __name__ == "__main__":
    print("Creating CAE MTL model...")
    
    # 방법 1: 직접 모델 생성
    model = create_cae_mtl(
        input_shape=(32, 32, 3),
        num_classes=10,
        recon_weight=0.7,
        cls_weight=0.3,
        recon_loss='mae'
    )
    
    print("\n" + "="*60)
    print("CAE MTL Architecture:")
    print("="*60)
    model.summary()
    
    # 방법 2: 래퍼 클래스 사용
    cae_mtl = CAEMTL(
        input_shape=(32, 32, 3),
        num_classes=10,
        recon_weight=0.7,
        cls_weight=0.3,
        recon_loss='mae',
        dropout_rate=0.1
    )
    
    print("\n" + "="*60)
    print("CAE MTL Wrapper initialized")
    print(f"Reconstruction loss: {cae_mtl.recon_loss.upper()}")
    print(f"Loss weights - Restoration: {cae_mtl.recon_weight:.3f}, "
          f"Classification: {cae_mtl.cls_weight:.3f}")
    print("="*60)