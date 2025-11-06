import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_cae_multitask(input_shape_img=(32, 32, 3), num_classes=10,
                       dropout_rate=0.1,
                       l2_regularization=1e-4) -> Model:
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
        l2_regularization: L2 regularization 강도
    """
    image_input = layers.Input(shape=input_shape_img, name='image_input')

    # =====================================
    # Encoder (4 stages)
    # =====================================
    encoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(image_input)
    encoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_1 = layers.MaxPooling2D(2)(encoder_layer)  # 16x16

    encoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(pooled_layer_1)
    encoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_2 = layers.MaxPooling2D(2)(encoder_layer)  # 8x8

    encoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(pooled_layer_2)
    encoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_3 = layers.MaxPooling2D(2)(encoder_layer)  # 4x4

    encoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(pooled_layer_3)
    encoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_4 = layers.MaxPooling2D(2)(encoder_layer)  # 2x2

    # =====================================
    # Bottleneck
    # =====================================
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(pooled_layer_4)
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(bottleneck)

    # =====================================
    # Classification Head (from bottleneck)
    # =====================================
    features = layers.GlobalAveragePooling2D()(bottleneck)
    features = layers.Dense(128, activation='relu', 
                       kernel_regularizer=regularizers.l2(l2_regularization))(features)
    features = layers.Dropout(dropout_rate)(features)  # ✅ 0.5 → 0.1
    classification_output = layers.Dense(num_classes, activation='softmax', 
                          name='classification_output')(features)

    # =====================================
    # Decoder (4 stages, symmetric)
    # NO skip connections - 순수 bottleneck 복원
    # =====================================
    decoder_layer = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bottleneck)  # 4x4
    decoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(decoder_layer)  # 8x8
    decoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(decoder_layer)  # 16x16
    decoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(decoder_layer)  # 32x32
    decoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_layer)

    # ✅ Sigmoid activation (픽셀 값 [0,1] 범위 보장)
    restoration_output = layers.Conv2D(3, 1, activation='sigmoid', 
                       name='restoration_output')(decoder_layer)
    
    return Model(inputs=image_input, outputs=[restoration_output, classification_output], name='CAE_multitask')


def build_cae_restoration(input_shape_img=(32, 32, 3)) -> Model:
    """
    Restoration-only CAE (초저 SNR 최적화, Skip connection 없음)
    
    Sequential BAM과 비교하기 위한 복원 전용 모델
    """
    image_input = layers.Input(shape=input_shape_img)

    # Encoder (4 stages)
    encoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(image_input)
    encoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_1 = layers.MaxPooling2D(2)(encoder_layer)  # 16x16

    encoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(pooled_layer_1)
    encoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_2 = layers.MaxPooling2D(2)(encoder_layer)  # 8x8

    encoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(pooled_layer_2)
    encoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_3 = layers.MaxPooling2D(2)(encoder_layer)  # 4x4

    encoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(pooled_layer_3)
    encoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(encoder_layer)
    pooled_layer_4 = layers.MaxPooling2D(2)(encoder_layer)  # 2x2

    # Bottleneck
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(pooled_layer_4)
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(bottleneck)

    # Decoder (4 stages)
    decoder_layer = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bottleneck)  # 4x4
    decoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(decoder_layer)  # 8x8
    decoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(decoder_layer)  # 16x16
    decoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_layer)

    decoder_layer = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(decoder_layer)  # 32x32
    decoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_layer)
    decoder_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_layer)

    # ✅ Sigmoid activation
    restoration_output = layers.Conv2D(3, 1, activation='sigmoid')(decoder_layer)
    return Model(inputs=image_input, outputs=restoration_output, name='CAE_restoration')


def build_cae_classification(input_shape_img=(32, 32, 3), num_classes=10,
                             dropout_rate=0.2) -> Model:
    """
    Classification-only CAE encoder
    
    Sequential BAM과 비교하기 위한 분류 전용 모델
    복원된 이미지를 입력으로 받아 분류
    """
    restored_image_input = layers.Input(shape=input_shape_img, name='restored_input')

    # Encoder (간단한 버전 - 이미 복원된 이미지 입력)
    encoder_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(restored_image_input)
    encoder_layer = layers.MaxPooling2D(2)(encoder_layer)  # 16x16

    encoder_layer = layers.Conv2D(128, 3, padding='same', activation='relu')(encoder_layer)
    encoder_layer = layers.MaxPooling2D(2)(encoder_layer)  # 8x8

    encoder_layer = layers.Conv2D(256, 3, padding='same', activation='relu')(encoder_layer)
    encoder_layer = layers.MaxPooling2D(2)(encoder_layer)  # 4x4

    encoder_layer = layers.Conv2D(512, 3, padding='same', activation='relu')(encoder_layer)
    encoder_layer = layers.GlobalAveragePooling2D()(encoder_layer)

    # Classifier
    features = layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(1e-4))(encoder_layer)
    features = layers.Dropout(dropout_rate)(features)
    classification_output = layers.Dense(num_classes, activation='softmax')(features)

    return Model(inputs=restored_image_input, outputs=classification_output, name='CAE_classification')


class SequentialCAE:
    """
    연쇄 CAE: 복원 → 분류
    
    Sequential BAM과 비교하기 위한 CNN 버전
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.restore_model = build_cae_restoration(input_shape)
        self.cls_model = build_cae_classification(input_shape, num_classes)
    
    def compile_models(self, restoration_learning_rate=1e-3, classification_learning_rate=1e-3, restoration_loss='mse'):
        """두 모델을 독립적으로 컴파일"""
        self.restore_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=restoration_learning_rate),
            loss=restoration_loss,  # 'mse' or 'mae'
            metrics=['mse', 'mae']
        )
        
        self.cls_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=classification_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_stage1(self, noisy_images, clean_images, epochs=50, batch_size=128,
                     validation_split=0.1, callbacks=None):
        """Stage 1: 복원 학습"""
        print("\n" + "="*60)
        print("Stage 1: Training CAE Restoration")
        print("="*60)
        
        history1 = self.restore_model.fit(
            noisy_images, clean_images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history1
    
    def train_stage2(self, noisy_images, labels, epochs=50, batch_size=128,
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
        
        num_samples = len(noisy_images)
        restored_images_list = []
        
        # 큰 청크로 나눠서 처리 (4배 배치 크기)
        chunk_size = batch_size * 4
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            batch_restored = self.restore_model.predict(
                noisy_images[start_idx:end_idx],
                batch_size=batch_size,
                verbose=0
            )
            restored_images_list.append(batch_restored)
            
            # 진행 상황 출력
            if (start_idx // chunk_size) % 10 == 0:
                print(f"  Processed {end_idx}/{num_samples} samples...")
        
        restored_images = np.concatenate(restored_images_list, axis=0)
        del restored_images_list  # 메모리 해제
        gc.collect()
        
        print(f"✓ Restored images shape: {restored_images.shape}")
        
        # 분류 학습
        print("\nTraining classification model...")
        history2 = self.cls_model.fit(
            restored_images, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history2
    
    def predict(self, noisy_images, batch_size=128):
        """연쇄 예측"""
        restored_images = self.restore_model.predict(noisy_images, batch_size=batch_size, verbose=0)
        predictions = self.cls_model.predict(restored_images, batch_size=batch_size, verbose=0)
        
        return {
            "restored": restored_images,
            "predictions": predictions
        }
    
    def evaluate(self, noisy_images, clean_images, labels, batch_size=128):
        """
        전체 파이프라인 평가
        
        Args:
            noisy_images: 노이즈가 추가된 입력
            clean_images: 깨끗한 이미지 (복원 평가용)
            labels: 원-핫 인코딩된 레이블
            batch_size: 배치 크기
            
        Returns:
            dict: 복원 MSE/MAE/PSNR와 분류 정확도
        """
        outputs = self.predict(noisy_images, batch_size=batch_size)
        
        # 복원 성능
        import numpy as np
        mean_squared_error = np.mean((outputs['restored'] - clean_images) ** 2)
        mean_absolute_error = np.mean(np.abs(outputs['restored'] - clean_images))
        psnr = 10 * np.log10(1.0 / mean_squared_error) if mean_squared_error > 0 else float('inf')
        
        # 분류 성능
        predicted_classes = np.argmax(outputs['predictions'], axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Top-3 accuracy
        top3_predictions = np.argsort(outputs['predictions'], axis=1)[:, -3:]
        top3_accuracy = np.mean([true_class in top3 for true_class, top3 in zip(true_classes, top3_predictions)])
        
        return {
            "restoration": {
                "mse": float(mean_squared_error),
                "mae": float(mean_absolute_error),
                "psnr": float(psnr)
            },
            "classification": {
                "accuracy": float(accuracy),
                "top3_accuracy": float(top3_accuracy)
            }
        }