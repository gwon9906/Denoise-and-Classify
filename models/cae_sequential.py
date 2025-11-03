import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_cae_multitask(input_shape_img=(32, 32, 3), num_classes=10,
                       dropout_rate=0.1,
                       l2_reg=1e-4) -> Model:
    """
    Multitask CAE (초저 SNR 최적화, Skip connection 없음)
    
    U-Net과의 비교를 위해 skip connection을 의도적으로 제거.
    순수하게 bottleneck을 통한 정보 압축/복원 능력 평가.
    
    초저 SNR 최적화:
    - 낮은 dropout (0.1 vs 0.5)
    - Sigmoid output activation
    - BatchNorm 추가 옵션
    
    Args:
        input_shape_img: 입력 이미지 shape (32, 32, 3)
        num_classes: 분류 클래스 수
        dropout_rate: Dropout 비율 (초저 SNR: 0.05~0.1)
        l2_reg: L2 regularization 강도
    """
    img_in = layers.Input(shape=input_shape_img, name='image_input')

    # =====================================
    # Encoder (4 stages)
    # =====================================
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    p1 = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    p2 = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    p3 = layers.MaxPooling2D(2)(x)  # 4x4

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    p4 = layers.MaxPooling2D(2)(x)  # 2x2

    # =====================================
    # Bottleneck
    # =====================================
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # =====================================
    # Classification Head (from bottleneck)
    # =====================================
    feat = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(128, activation='relu', 
                       kernel_regularizer=regularizers.l2(l2_reg))(feat)
    feat = layers.Dropout(dropout_rate)(feat)  # ✅ 0.5 → 0.1
    cls_out = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(feat)

    # =====================================
    # Decoder (4 stages, symmetric)
    # NO skip connections - 순수 bottleneck 복원
    # =====================================
    d = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # 4x4
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d)  # 8x8
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d)  # 16x16
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)  # 32x32
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    # ✅ Sigmoid activation (픽셀 값 [0,1] 범위 보장)
    rec = layers.Conv2D(3, 1, activation='sigmoid', 
                       name='restoration_output')(d)
    
    return Model(inputs=img_in, outputs=[rec, cls_out], name='CAE_multitask')


def build_cae_restoration(input_shape_img=(32, 32, 3)) -> Model:
    """
    Restoration-only CAE (초저 SNR 최적화, Skip connection 없음)
    
    Sequential BAM과 비교하기 위한 복원 전용 모델
    """
    img_in = layers.Input(shape=input_shape_img)

    # Encoder (4 stages)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    p1 = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    p2 = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    p3 = layers.MaxPooling2D(2)(x)  # 4x4

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    p4 = layers.MaxPooling2D(2)(x)  # 2x2

    # Bottleneck
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # Decoder (4 stages)
    d = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # 4x4
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(256, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d)  # 8x8
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(128, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d)  # 16x16
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(64, 3, padding='same', activation='relu')(d)

    d = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d)  # 32x32
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    # ✅ Sigmoid activation
    rec = layers.Conv2D(3, 1, activation='sigmoid')(d)
    return Model(inputs=img_in, outputs=rec, name='CAE_restoration')


def build_cae_classification(input_shape_img=(32, 32, 3), num_classes=10,
                             dropout_rate=0.2) -> Model:
    """
    Classification-only CAE encoder
    
    Sequential BAM과 비교하기 위한 분류 전용 모델
    복원된 이미지를 입력으로 받아 분류
    """
    img_in = layers.Input(shape=input_shape_img, name='restored_input')

    # Encoder (간단한 버전 - 이미 복원된 이미지 입력)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(img_in)
    x = layers.MaxPooling2D(2)(x)  # 16x16

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)  # 8x8

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)  # 4x4

    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    cls_out = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=img_in, outputs=cls_out, name='CAE_classification')


class SequentialCAE:
    """
    연쇄 CAE: 복원 → 분류
    
    Sequential BAM과 비교하기 위한 CNN 버전
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.restore_model = build_cae_restoration(input_shape)
        self.cls_model = build_cae_classification(input_shape, num_classes)
    
    def compile_models(self, restore_lr=1e-3, cls_lr=1e-3, restore_loss='mse'):
        """두 모델을 독립적으로 컴파일"""
        self.restore_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=restore_lr),
            loss=restore_loss,  # 'mse' or 'mae'
            metrics=['mse', 'mae']
        )
        
        self.cls_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cls_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_stage1(self, x_noisy, x_clean, epochs=50, batch_size=128,
                     validation_split=0.1, callbacks=None):
        """Stage 1: 복원 학습"""
        print("\n" + "="*60)
        print("Stage 1: Training CAE Restoration")
        print("="*60)
        
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
        """Stage 2: 분류 학습"""
        # 메모리 정리
        import gc
        gc.collect()
        
        print("\n" + "="*60)
        print("Stage 2: Training CAE Classification")
        print("="*60)
        
        # ✅ 배치 단위로 복원 (메모리 효율)
        print("Generating restored images for training...")
        import numpy as np
        
        n_samples = len(x_noisy)
        x_restored_list = []
        
        # 큰 청크로 나눠서 처리 (4배 배치 크기)
        chunk_size = batch_size * 4
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            x_batch_restored = self.restore_model.predict(
                x_noisy[i:end_idx],
                batch_size=batch_size,
                verbose=0
            )
            x_restored_list.append(x_batch_restored)
            
            # 진행 상황 출력
            if (i // chunk_size) % 10 == 0:
                print(f"  Processed {end_idx}/{n_samples} samples...")
        
        x_restored = np.concatenate(x_restored_list, axis=0)
        del x_restored_list  # 메모리 해제
        gc.collect()
        
        print(f"✓ Restored images shape: {x_restored.shape}")
        
        # 분류 학습
        print("\nTraining classification model...")
        history2 = self.cls_model.fit(
            x_restored, y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history2
    
    def predict(self, x_noisy, batch_size=128):
        """연쇄 예측"""
        x_restored = self.restore_model.predict(x_noisy, batch_size=batch_size, verbose=0)
        y_pred = self.cls_model.predict(x_restored, batch_size=batch_size, verbose=0)
        
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
            dict: 복원 MSE/MAE/PSNR와 분류 정확도
        """
        outputs = self.predict(x_noisy, batch_size=batch_size)
        
        # 복원 성능
        import numpy as np
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