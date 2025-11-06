# Sequential MF-BAM 모델 (연쇄 학습)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .bam_mf import MF, build_mf_module, build_decoder
from .bam import BAM, BAMAssociativeLayer


class SequentialBAM:
    """
    Sequential BAM: MF(복원) → BAM(분류)
    
    연쇄형 학습 절차:
    1. Stage 1 (MF 학습): x_noisy → x_restored (MAE 손실)
       - 손상된 입력을 복원하도록 MF 모듈 학습
       
    2. Stage 2 (BAM 학습): x_restored → y_bipolar
       - 복원된 이미지로부터 분류
       - BAMAssociativeLayer 사용 (반복 안정화)
    
    추론:
    - x_noisy → MF → x_restored → BAM → y_pred
    """
    
    def __init__(self, input_dim=3072, denoise_latent=256, cls_latent=None, 
                 num_classes=10, bam_steps=5, bam_temperature=0.5):
        """
        Args:
            input_dim: 입력 차원 (32*32*3 = 3072)
            denoise_latent: MF 잠재 차원
            cls_latent: 사용 안함 (호환성 유지)
            num_classes: 분류 클래스 수
            bam_steps: BAM 반복 횟수
            bam_temperature: BAM temperature
        """
        self.input_dim = input_dim
        self.denoise_latent = denoise_latent
        self.num_classes = num_classes
        self.bam_steps = bam_steps
        self.bam_temperature = bam_temperature
        
        # MF 모듈 생성 (복원용)
        encoder_dims = [1024, 768, 512, denoise_latent]
        self.multifactor_module = MF(input_dim=input_dim, encoder_dims=encoder_dims)
        
        # Keras 모델로 래핑
        self._build_keras_models()
        
    def _build_keras_models(self):
        """MF와 BAM을 Keras 모델로 래핑"""
        # Denoising 모델 (MF)
        noisy_input = layers.Input(shape=(self.input_dim,), name='noisy_input')
        restored_output, _ = self.multifactor_module(noisy_input)
        self.denoise_model = keras.Model(
            inputs=noisy_input, 
            outputs=restored_output, 
            name='denoise_bam'
        )
        
        # Classification 모델 (BAM)
        denoised_input = layers.Input(shape=(self.input_dim,), name='denoised_input')
        
        # BAMAssociativeLayer 사용
        classification_logits = BAMAssociativeLayer(
            input_dim=self.input_dim,
            output_dim=self.num_classes,
            steps=self.bam_steps,
            temperature=self.bam_temperature,
            name='bam_assoc'
        )(denoised_input)
        
        classification_output = layers.Activation('softmax', name='classification_output')(classification_logits)
        
        self.classification_model = keras.Model(
            inputs=denoised_input,
            outputs=classification_output,
            name='classification_bam'
        )
    
    def compile_models(self, denoise_learning_rate=1e-3, classification_learning_rate=2e-4):
        """
        두 모델을 독립적으로 컴파일
        
        Args:
            denoise_learning_rate: MF 학습률
            classification_learning_rate: BAM 학습률
        """
        # Stage 1: MF 복원 모델 (MAE 손실)
        # ✅ Gradient clipping으로 안정성 확보 (NaN 방지)
        self.denoise_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=denoise_learning_rate,
                clipnorm=1.0,      # gradient norm clipping
                clipvalue=0.5      # gradient value clipping
            ),
            loss='mae',  # MAE 손실 사용
            metrics=[
                keras.metrics.MeanSquaredError(name="mse"),
                keras.metrics.MeanAbsoluteError(name="mae")
            ]
        )
        
        # Stage 2: BAM 분류 모델
        self.classification_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=classification_learning_rate,
                clipnorm=1.0
            ),
            loss='categorical_crossentropy',
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
            ]
        )
        
        print("✓ Models compiled successfully (with gradient clipping)")
    
    def train_stage1(self, noisy_images, clean_images, epochs=50, batch_size=128,
                     validation_split=0.2, callbacks=None):
        """
        Stage 1: MF 복원 학습
        
        Args:
            noisy_images: 노이즈가 추가된 입력 (flattened)
            clean_images: 깨끗한 타겟 (flattened)
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        print("\n" + "="*60)
        print("Stage 1: Training Restoration MF")
        print("="*60)
        print(f"Architecture: MF (Multifactor) Autoencoder")
        print(f"Input dim: {self.input_dim}")
        print(f"Latent dim: {self.denoise_latent}")
        print(f"Loss: MAE (Mean Absolute Error)")
        print(f"Dropout: None (극저 SNR 최적화)")
        
        history_stage1 = self.denoise_model.fit(
            noisy_images, clean_images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Stage 1 training complete!")
        return history_stage1
    
    def train_stage2(self, noisy_images, labels, epochs=50, batch_size=128,
                     validation_split=0.2, callbacks=None):
        """
        Stage 2: BAM 분류 학습
        
        학습된 MF 모듈로 훈련 입력을 복원한 뒤,
        복원된 이미지를 입력으로 BAM 분류기를 학습
        
        Args:
            noisy_images: 노이즈가 추가된 입력 (복원용)
            labels: 원-핫 인코딩된 레이블
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        # 메모리 정리 (모델은 유지)
        import gc
        gc.collect()
        
        print("\n" + "="*60)
        print("Stage 2: Training Classification BAM")
        print("="*60)
        print(f"BAM input dim: {self.input_dim}")
        print(f"BAM output dim: {self.num_classes}")
        print(f"BAM steps: {self.bam_steps}")
        print(f"BAM temperature: {self.bam_temperature}")
        
        # 1. MF로 훈련 데이터 복원 (배치 단위로 처리 - 메모리 효율)
        print("Generating denoised images for training...")
        
        num_samples = len(noisy_images)
        restored_images_list = []
        
        # 배치 단위로 복원 (메모리 절약)
        for start_idx in range(0, num_samples, batch_size * 4):  # 4배 큰 청크로
            end_idx = min(start_idx + batch_size * 4, num_samples)
            batch_restored = self.denoise_model.predict(
                noisy_images[start_idx:end_idx],
                batch_size=batch_size,
                verbose=0
            )
            restored_images_list.append(batch_restored)
            
            if (start_idx // (batch_size * 4)) % 10 == 0:
                print(f"  Processed {end_idx}/{num_samples} samples...")
        
        import numpy as np
        restored_images = np.concatenate(restored_images_list, axis=0)
        del restored_images_list  # 메모리 해제
        
        print(f"✓ Restored images shape: {restored_images.shape}")
        
        # 2. BAM 학습 (Keras fit 사용)
        print("\nTraining BAM with softmax + categorical crossentropy")
        history_stage2 = self.classification_model.fit(
            restored_images, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Stage 2 training complete!")
        return history_stage2
    
    def predict(self, noisy_images, batch_size=128, verbose=0):
        """
        연쇄 추론: x_noisy → MF → x_restored → BAM → y_pred
        
        Args:
            noisy_images: 노이즈가 추가된 입력
            batch_size: 배치 크기
            verbose: 진행 표시 레벨
            
        Returns:
            dict: {
                "denoised": 복원된 이미지,
                "predictions": 클래스 확률
            }
        """
        # Stage 1: MF로 복원
        restored_images = self.denoise_model.predict(
            noisy_images,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Stage 2: BAM으로 분류
        predictions = self.classification_model.predict(
            restored_images,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return {
            "denoised": restored_images,
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
        outputs = self.predict(noisy_images, batch_size=batch_size, verbose=1)
        
        # 복원 성능
        mean_squared_error = np.mean((outputs['denoised'] - clean_images) ** 2)
        mean_absolute_error = np.mean(np.abs(outputs['denoised'] - clean_images))
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
    
    def save_models(self, denoise_path="bam_mf_denoise.keras",
                    cls_path="bam_classification.keras"):
        """두 모델을 개별적으로 저장"""
        self.denoise_model.save(denoise_path)
        self.cls_model.save(cls_path)
        print(f"✓ Models saved: {denoise_path}, {cls_path}")
    
    def load_models(self, denoise_path="bam_mf_denoise.keras",
                    cls_path="bam_classification.keras"):
        """저장된 모델 로드"""
        self.denoise_model = keras.models.load_model(denoise_path)
        self.cls_model = keras.models.load_model(cls_path)
        print(f"✓ Models loaded: {denoise_path}, {cls_path}")


# ============================================
# 편의 함수
# ============================================

def create_sequential_bam(input_dim=3072, denoise_latent=256, num_classes=10,
                          denoise_learning_rate=1e-3, classification_learning_rate=2e-4,
                          bam_steps=5, bam_temperature=0.5):
    """
    Sequential BAM 생성 및 컴파일 (one-liner)
    
    Returns:
        컴파일된 SequentialBAM 인스턴스
    """
    sequential_bam = SequentialBAM(
        input_dim=input_dim,
        denoise_latent=denoise_latent,
        num_classes=num_classes,
        bam_steps=bam_steps,
        bam_temperature=bam_temperature
    )
    sequential_bam.compile_models(
        denoise_learning_rate=denoise_learning_rate,
        classification_learning_rate=classification_learning_rate
    )
    return sequential_bam


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Creating Sequential BAM (MF + BAM)...")
    
    # 모델 생성
    seq_bam = create_sequential_bam(
        input_dim=3072,
        denoise_latent=256,
        num_classes=10,
        bam_steps=5,
        bam_temp=0.5
    )
    
    # 모델 구조 확인
    print("\n" + "="*60)
    print("Denoising Model (MF) Architecture:")
    print("="*60)
    seq_bam.denoise_model.summary()
    
    print("\n" + "="*60)
    print("Classification Model (BAM) Architecture:")
    print("="*60)
    seq_bam.cls_model.summary()
    
    print("\n" + "="*60)
    print("Sequential BAM initialized")
    print("Stage 1: MF with MAE loss (No Dropout)")
    print("Stage 2: BAMAssociativeLayer with iterative recall")
    print("="*60)

