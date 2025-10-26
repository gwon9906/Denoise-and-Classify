# Sequential U-Net 모델 (복원 → 분류)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import numpy as np


# ============================================
# 1. 복원 전용 U-Net
# ============================================

def build_unet_restoration(input_shape=(32, 32, 3), dropout_rate=0.0, l2_reg=1e-5):
    """
    Restoration-only U-Net (초저 SNR 최적화)
    
    특징:
    - Skip connections (공간 정보 보존)
    - Residual learning (노이즈 예측)
    - BatchNormalization
    
    Args:
        input_shape: 입력 이미지 shape (H, W, C)
        dropout_rate: Dropout 비율 (초저 SNR: 0.0~0.05)
        l2_reg: L2 regularization 강도
        
    Returns:
        복원 전용 U-Net 모델
    """
    
    def conv_block(x, nf, name_prefix):
        """기본 컨볼루션 블록"""
        x = layers.Conv2D(nf, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        x = layers.Conv2D(nf, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        return x

    # Input
    image_input = layers.Input(shape=input_shape, name="noisy_input")

    # =====================================
    # Encoder (4 layers)
    # =====================================
    c1 = conv_block(image_input, 32, 'enc1')  # 32x32
    p1 = layers.MaxPooling2D(2, name='pool1')(c1)
    
    c2 = conv_block(p1, 64, 'enc2')  # 16x16
    p2 = layers.MaxPooling2D(2, name='pool2')(c2)
    
    c3 = conv_block(p2, 128, 'enc3')  # 8x8
    p3 = layers.MaxPooling2D(2, name='pool3')(c3)
    
    c4 = conv_block(p3, 256, 'enc4')  # 4x4
    p4 = layers.MaxPooling2D(2, name='pool4')(c4)

    # =====================================
    # Bottleneck
    # =====================================
    b = conv_block(p4, 512, 'bottleneck')  # 2x2

    # =====================================
    # Decoder (4 layers) with skip connections
    # =====================================
    d4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same',
                               name='upsample4')(b)  # 4x4
    d4 = layers.Concatenate(name='concat4')([d4, c4])
    d4 = conv_block(d4, 256, 'dec4')

    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same',
                               name='upsample3')(d4)  # 8x8
    d3 = layers.Concatenate(name='concat3')([d3, c3])
    d3 = conv_block(d3, 128, 'dec3')

    d2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same',
                               name='upsample2')(d3)  # 16x16
    d2 = layers.Concatenate(name='concat2')([d2, c2])
    d2 = conv_block(d2, 64, 'dec2')

    d1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same',
                               name='upsample1')(d2)  # 32x32
    d1 = layers.Concatenate(name='concat1')([d1, c1])
    d1 = conv_block(d1, 32, 'dec1')

    # =====================================
    # Restoration Output (Residual Learning)
    # =====================================
    # 노이즈(residual)를 예측하고, input에서 빼서 복원
    residual_pred = layers.Conv2D(3, 1, activation='linear',
                                  name="residual_pred")(d1)
    restoration_output = layers.Subtract(name="restored_output")(
        [image_input, residual_pred]
    )

    model = Model(
        inputs=image_input,
        outputs=restoration_output,
        name='UNet_Restoration'
    )
    return model


# ============================================
# 2. 분류 전용 U-Net Encoder
# ============================================

def build_unet_classification(input_shape=(32, 32, 3), num_classes=10,
                              dropout_rate=0.1, l2_reg=1e-4):
    """
    Classification-only U-Net Encoder
    
    복원된 이미지를 입력으로 받아 분류
    U-Net encoder + multi-scale feature aggregation
    
    Args:
        input_shape: 입력 이미지 shape (H, W, C)
        num_classes: 분류 클래스 수
        dropout_rate: Dropout 비율
        l2_reg: L2 regularization 강도
        
    Returns:
        분류 전용 모델
    """
    
    def conv_block(x, nf, name_prefix):
        """기본 컨볼루션 블록"""
        x = layers.Conv2D(nf, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        x = layers.Conv2D(nf, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        return x

    # Input (복원된 이미지)
    image_input = layers.Input(shape=input_shape, name="restored_input")

    # =====================================
    # Encoder (multi-scale feature extraction)
    # =====================================
    c1 = conv_block(image_input, 32, 'cls_enc1')  # 32x32
    p1 = layers.MaxPooling2D(2, name='cls_pool1')(c1)
    
    c2 = conv_block(p1, 64, 'cls_enc2')  # 16x16
    p2 = layers.MaxPooling2D(2, name='cls_pool2')(c2)
    
    c3 = conv_block(p2, 128, 'cls_enc3')  # 8x8
    p3 = layers.MaxPooling2D(2, name='cls_pool3')(c3)
    
    c4 = conv_block(p3, 256, 'cls_enc4')  # 4x4
    p4 = layers.MaxPooling2D(2, name='cls_pool4')(c4)

    # Bottleneck
    b = conv_block(p4, 512, 'cls_bottleneck')  # 2x2

    # =====================================
    # Multi-scale Feature Aggregation
    # =====================================
    gap_b = layers.GlobalAveragePooling2D(name='cls_gap_b')(b)
    gap_c3 = layers.GlobalAveragePooling2D(name='cls_gap_c3')(c3)
    gap_c1 = layers.GlobalAveragePooling2D(name='cls_gap_c1')(c1)
    
    feat = layers.Concatenate(name='cls_concat')([gap_b, gap_c3, gap_c1])

    # =====================================
    # Classifier
    # =====================================
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='cls_dense1')(feat)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='cls_dropout')(x)
    
    classification_output = layers.Dense(num_classes, activation='softmax',
                                        name="classification_output")(x)

    model = Model(
        inputs=image_input,
        outputs=classification_output,
        name='UNet_Classification'
    )
    return model


# ============================================
# 3. Sequential U-Net 래퍼 클래스
# ============================================

class SequentialUNet:
    """
    Sequential U-Net: 복원 → 분류
    
    두 개의 독립적인 U-Net을 순차적으로 연결:
    1. Restoration U-Net: x' → x̂ (노이즈 제거)
    2. Classification U-Net: x̂ → ŷ (분류)
    
    장점:
    - 각 태스크에 특화된 학습
    - Skip connection으로 최고의 복원 성능
    - Residual learning
    
    단점:
    - 2-pass 추론 (느림)
    - 오류 누적 가능성
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10,
                 restore_dropout=0.0, cls_dropout=0.1):
        """
        Args:
            input_shape: 입력 이미지 shape
            num_classes: 분류 클래스 수
            restore_dropout: 복원 모델 dropout (초저 SNR: 0.0)
            cls_dropout: 분류 모델 dropout
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.restore_dropout = restore_dropout
        self.cls_dropout = cls_dropout
        
        # 두 개의 독립적인 모델 생성
        self.restore_model = build_unet_restoration(
            input_shape,
            dropout_rate=restore_dropout
        )
        self.cls_model = build_unet_classification(
            input_shape,
            num_classes,
            dropout_rate=cls_dropout
        )
    
    def compile_models(self, restore_lr=1e-3, cls_lr=1e-3, restore_loss='mae'):
        """
        두 모델을 독립적으로 컴파일
        
        Args:
            restore_lr: Restoration U-Net 학습률
            cls_lr: Classification U-Net 학습률
            restore_loss: 'mae', 'mse', 'huber'
        """
        # Stage 1: Restoration
        if restore_loss == 'mae':
            loss_fn = keras.losses.MeanAbsoluteError()
        elif restore_loss == 'mse':
            loss_fn = keras.losses.MeanSquaredError()
        elif restore_loss == 'huber':
            loss_fn = keras.losses.Huber(delta=0.1)
        else:
            raise ValueError(f"Unknown loss: {restore_loss}")
        
        self.restore_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=restore_lr),
            loss=loss_fn,
            metrics=[
                keras.metrics.MeanSquaredError(name="mse"),
                keras.metrics.MeanAbsoluteError(name="mae")
            ]
        )
        
        # Stage 2: Classification
        self.cls_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cls_lr),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="acc"),
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
            ]
        )
    
    def train_stage1(self, x_noisy, x_clean, epochs=50, batch_size=128,
                     validation_split=0.1, callbacks=None):
        """
        Stage 1: 복원 모델 학습
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            x_clean: 깨끗한 타겟
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        print("\n" + "="*60)
        print("Stage 1: Training Restoration U-Net")
        print("="*60)
        print(f"Architecture: U-Net with Skip Connections + Residual Learning")
        print(f"Dropout rate: {self.restore_dropout}")
        
        history1 = self.restore_model.fit(
            x_noisy, x_clean,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history1
    
    def train_stage2(self, x_noisy, y_labels, epochs=50, batch_size=128,
                     validation_split=0.1, callbacks=None):
        """
        Stage 2: 분류 모델 학습
        복원된 이미지를 사용하여 학습
        
        Args:
            x_noisy: 노이즈가 추가된 입력 (복원용)
            y_labels: 원-핫 인코딩된 레이블
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        print("\n" + "="*60)
        print("Stage 2: Training Classification U-Net")
        print("="*60)
        print(f"Dropout rate: {self.cls_dropout}")
        
        # Stage 1으로 노이즈 제거
        print("Restoring training data...")
        x_restored = self.restore_model.predict(
            x_noisy,
            batch_size=batch_size,
            verbose=1
        )
        
        # 복원된 데이터로 분류 모델 학습
        history2 = self.cls_model.fit(
            x_restored, y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history2
    
    def predict(self, x_noisy, batch_size=128, verbose=0):
        """
        연쇄 추론: x' → x̂ → ŷ
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            batch_size: 배치 크기
            verbose: 진행 표시 레벨
            
        Returns:
            dict: {
                "restored": 복원된 이미지,
                "predictions": 클래스 확률
            }
        """
        # Stage 1: 복원
        x_restored = self.restore_model.predict(
            x_noisy,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Stage 2: 분류
        y_pred = self.cls_model.predict(
            x_restored,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return {
            "restored": x_restored,
            "predictions": y_pred
        }
    
    def evaluate(self, x_noisy, x_clean, y_labels, batch_size=128):
        """
        전체 파이프라인 평가
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            x_clean: 깨끗한 이미지 (복원 평가용)
            y_labels: 원-핫 인코딩된 레이블
            batch_size: 배치 크기
            
        Returns:
            dict: 복원 MSE/MAE와 분류 정확도
        """
        outputs = self.predict(x_noisy, batch_size=batch_size, verbose=1)
        
        # 복원 성능
        mse = np.mean((outputs['restored'] - x_clean) ** 2)
        mae = np.mean(np.abs(outputs['restored'] - x_clean))
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        # 분류 성능
        y_pred_classes = np.argmax(outputs['predictions'], axis=1)
        y_true_classes = np.argmax(y_labels, axis=1)
        accuracy = np.mean(y_pred_classes == y_true_classes)
        
        # Top-3 accuracy
        top3_preds = np.argsort(outputs['predictions'], axis=1)[:, -3:]
        top3_acc = np.mean([y_true in top3 for y_true, top3 in zip(y_true_classes, top3_preds)])
        
        return {
            "restoration": {
                "mse": float(mse),
                "mae": float(mae),
                "psnr": float(psnr)
            },
            "classification": {
                "accuracy": float(accuracy),
                "top3_accuracy": float(top3_acc)
            }
        }
    
    def save_models(self, restore_path="unet_restore.keras",
                    cls_path="unet_classification.keras"):
        """
        두 모델을 개별적으로 저장
        """
        self.restore_model.save(restore_path)
        self.cls_model.save(cls_path)
        print(f"Models saved: {restore_path}, {cls_path}")
    
    def load_models(self, restore_path="unet_restore.keras",
                    cls_path="unet_classification.keras"):
        """
        저장된 모델 로드
        """
        self.restore_model = keras.models.load_model(restore_path)
        self.cls_model = keras.models.load_model(cls_path)
        print(f"Models loaded: {restore_path}, {cls_path}")


# ============================================
# 4. 편의 함수
# ============================================

def create_sequential_unet(input_shape=(32, 32, 3), num_classes=10,
                           restore_lr=1e-3, cls_lr=1e-3,
                           restore_loss='mae',
                           restore_dropout=0.0, cls_dropout=0.1):
    """
    Sequential U-Net 생성 및 컴파일 (one-liner)
    
    Returns:
        컴파일된 SequentialUNet 인스턴스
    """
    seq_unet = SequentialUNet(
        input_shape=input_shape,
        num_classes=num_classes,
        restore_dropout=restore_dropout,
        cls_dropout=cls_dropout
    )
    seq_unet.compile_models(
        restore_lr=restore_lr,
        cls_lr=cls_lr,
        restore_loss=restore_loss
    )
    return seq_unet


# ============================================
# 5. 사용 예제
# ============================================

if __name__ == "__main__":
    print("Creating Sequential U-Net models...")
    
    # 모델 생성
    seq_unet = create_sequential_unet(
        input_shape=(32, 32, 3),
        num_classes=10,
        restore_loss='mae',
        restore_dropout=0.0,
        cls_dropout=0.1
    )
    
    # 모델 구조 확인
    print("\n" + "="*60)
    print("Restoration U-Net Architecture:")
    print("="*60)
    seq_unet.restore_model.summary()
    
    print("\n" + "="*60)
    print("Classification U-Net Architecture:")
    print("="*60)
    seq_unet.cls_model.summary()
    
    print("\n" + "="*60)
    print("Sequential U-Net initialized")
    print("Stage 1: Restoration with Skip Connections + Residual Learning")
    print("Stage 2: Classification with Multi-scale Features")
    print("="*60)