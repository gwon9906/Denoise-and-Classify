# U-Net MTL 모델
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import numpy as np


def build_multitask_unet(input_shape_img=(32, 32, 3), num_classes=10,
                        dropout_rate=0.0, l2_reg=1e-5):
    """
    Multi-task U-Net (초저 SNR 최적화)
    
    특징:
    1. Skip connections (공간 정보 보존)
    2. Residual learning (노이즈 예측)
    3. Multi-scale classification
    4. BatchNormalization
    
    Args:
        input_shape_img: 입력 이미지 shape (H, W, C)
        num_classes: 분류 클래스 수
        dropout_rate: Dropout 비율 (초저 SNR: 0.0~0.05)
        l2_reg: L2 regularization 강도
        
    Returns:
        Multi-output model: [restoration_output, classification_output]
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

    # =====================================
    # Input
    # =====================================
    image_input = layers.Input(shape=input_shape_img, name="image_input")

    # =====================================
    # Encoder (4 layers) with skip connections
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
    # Task 1: Restoration Head (Residual Learning)
    # =====================================
    # 노이즈(residual)를 예측하고, input에서 빼서 복원
    residual_pred = layers.Conv2D(3, 1, activation='linear', 
                                  name="residual_pred")(d1)
    restoration_output = layers.Subtract(name="restoration_output")(
        [image_input, residual_pred]
    )

    # =====================================
    # Task 2: Classification Head (Multi-scale)
    # =====================================
    # 다양한 스케일의 특징 결합
    gap_b = layers.GlobalAveragePooling2D(name='gap_bottleneck')(b)
    gap_d3 = layers.GlobalAveragePooling2D(name='gap_d3')(d3)
    gap_d1 = layers.GlobalAveragePooling2D(name='gap_d1')(d1)
    
    feat = layers.Concatenate(name='multiscale_concat')([gap_b, gap_d3, gap_d1])
    
    # Classifier
    cls_dense = layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4),
                            name='cls_dense1')(feat)
    
    if dropout_rate > 0:
        cls_dense = layers.Dropout(dropout_rate, name='cls_dropout')(cls_dense)
    
    classification_output = layers.Dense(num_classes, activation='softmax',
                                        name="classification_output")(cls_dense)

    model = Model(
        inputs=image_input,
        outputs=[restoration_output, classification_output],
        name='UNet_MTL'
    )
    return model


def create_unet_mtl(input_shape=(32, 32, 3), num_classes=10,
                    recon_weight=0.6, cls_weight=0.4, learning_rate=1e-3,
                    recon_loss='mae', dropout_rate=0.0):
    """
    U-Net MTL 생성 및 컴파일
    
    Args:
        input_shape: 입력 이미지 shape
        num_classes: 분류할 클래스 수
        recon_weight: restoration loss weight (0~1)
        cls_weight: classification loss weight (0~1)
        learning_rate: 학습률
        recon_loss: 'mae', 'mse', 'huber'
        dropout_rate: Dropout 비율
        
    Returns:
        컴파일된 U-Net MTL 모델
    """
    # Weight 정규화
    total_weight = recon_weight + cls_weight
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
        recon_weight = recon_weight / total_weight
        cls_weight = cls_weight / total_weight
        print(f"Normalized weights - Recon: {recon_weight:.3f}, Cls: {cls_weight:.3f}")
    
    model = build_multitask_unet(input_shape, num_classes, dropout_rate)
    
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
        "classification_output": keras.losses.CategoricalCrossentropy()  # ✅ 통일
    }
    
    loss_weights = {
        "restoration_output": recon_weight,
        "classification_output": cls_weight
    }
    
    metrics = {
        "restoration_output": [
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ],
        "classification_output": [
            keras.metrics.CategoricalAccuracy(name="acc"),  # ✅ 통일
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    }
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model


# ============================================
# U-Net MTL 래퍼 클래스
# ============================================

class UNetMTL:
    """
    U-Net MTL 래퍼 클래스
    
    Skip connection + Residual learning을 활용한 MTL
    - 최고의 공간 정보 보존
    - Multi-scale classification
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10,
                 recon_weight=0.6, cls_weight=0.4, learning_rate=1e-3,
                 recon_loss='mae', dropout_rate=0.0):
        """
        Args:
            input_shape: 입력 이미지 shape
            num_classes: 분류할 클래스 수
            recon_weight: restoration loss weight (0~1)
            cls_weight: classification loss weight (0~1)
            learning_rate: 학습률
            recon_loss: 'mae', 'mse', 'huber'
            dropout_rate: Dropout 비율 (초저 SNR에서는 0.0 권장)
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
        self.model = create_unet_mtl(
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
        U-Net MTL 학습
        
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
        print("Training U-Net MTL (Multi-Task Learning)")
        print("="*60)
        print(f"Architecture: U-Net with Skip Connections + Residual Learning")
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
        """U-Net MTL 예측"""
        outputs = self.model.predict(x_noisy, batch_size=batch_size, verbose=verbose)
        
        return {
            "restoration": outputs[0],
            "classification": outputs[1]
        }
    
    def evaluate(self, x_noisy, x_clean, y_labels, batch_size=128):
        """U-Net MTL 평가"""
        print("\n" + "="*60)
        print("Evaluating U-Net MTL")
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
        """상세한 평가"""
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
    
    def save_model(self, filepath="unet_mtl.keras"):
        """모델 저장"""
        self.model.save(filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath="unet_mtl.keras"):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Creating U-Net MTL model...")
    
    # 방법 1: 직접 모델 생성
    model = create_unet_mtl(
        input_shape=(32, 32, 3),
        num_classes=10,
        recon_weight=0.6,
        cls_weight=0.4,
        recon_loss='mae'
    )
    
    print("\n" + "="*60)
    print("U-Net MTL Architecture:")
    print("="*60)
    model.summary()
    
    # 방법 2: 래퍼 클래스 사용
    unet_mtl = UNetMTL(
        input_shape=(32, 32, 3),
        num_classes=10,
        recon_weight=0.6,
        cls_weight=0.4,
        recon_loss='mae',
        dropout_rate=0.0
    )
    
    print("\n" + "="*60)
    print("U-Net MTL Wrapper initialized")
    print(f"Skip Connections: ✅ Enabled")
    print(f"Residual Learning: ✅ Enabled")
    print(f"Multi-scale Classification: ✅ Enabled")
    print(f"Reconstruction loss: {unet_mtl.recon_loss.upper()}")
    print(f"Loss weights - Restoration: {unet_mtl.recon_weight:.3f}, "
          f"Classification: {unet_mtl.cls_weight:.3f}")
    print("="*60)