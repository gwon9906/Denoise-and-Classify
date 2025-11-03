"""
BAM Sequential Model v2 - 개선된 분류 네트워크
- Stage 1: MF (Denoising) - 기존과 동일
- Stage 2: Deep Classification Network (4층) + BAM - ⭐ 개선!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from .bam_mf import MF
from .bam import BAMAssociativeLayer


class SequentialBAMv2:
    """
    2단계 Sequential BAM 모델 (개선 버전)
    
    개선 사항:
    - Classification에 deep network 추가 (4층)
    - 점진적 차원 축소: 3072 → 1024 → 512 → 256 → 128
    - 최종적으로 BAM으로 128 → 10 분류
    """
    
    def __init__(self, input_dim=3072, denoise_latent=256, 
                 cls_hidden_dims=[1024, 512, 256, 128],
                 num_classes=10, bam_steps=5, bam_temp=0.5):
        """
        Args:
            input_dim: 입력 차원 (CIFAR-10: 3072)
            denoise_latent: MF 잠재 차원
            cls_hidden_dims: 분류기 hidden layer 차원들 (NEW!)
            num_classes: 클래스 개수
            bam_steps: BAM 반복 횟수
            bam_temp: BAM temperature
        """
        self.input_dim = input_dim
        self.denoise_latent = denoise_latent
        self.cls_hidden_dims = cls_hidden_dims
        self.num_classes = num_classes
        self.bam_steps = bam_steps
        self.bam_temp = bam_temp
        
        # Stage 1: Denoising MF (기존과 동일)
        self._build_denoise_model()
        
        # Stage 2: Deep Classification Network (개선!)
        self._build_classification_model()
    
    def _build_denoise_model(self):
        """Stage 1: MF 복원 모델 (기존과 동일)"""
        noisy_input = layers.Input(shape=(self.input_dim,), name='noisy_input')
        
        # MF 모듈 생성
        mf = MF(
            input_dim=self.input_dim,
            encoder_dims=[1024, 768, 512, self.denoise_latent]
        )
        
        # 복원된 이미지와 잠재 표현
        denoised, latent = mf(noisy_input)
        
        self.denoise_model = keras.Model(
            inputs=noisy_input,
            outputs=denoised,
            name='denoise_bam'
        )
    
    def _build_classification_model(self):
        """
        Stage 2: Deep Classification Network (개선!)
        
        구조:
        3072 → [Dense 1024 + BN + ReLU + Dropout]
             → [Dense 512 + BN + ReLU + Dropout]
             → [Dense 256 + BN + ReLU + Dropout]
             → [Dense 128 + BN + ReLU]
             → [BAM 128 → 10]
             → Softmax
        """
        denoised_input = layers.Input(shape=(self.input_dim,), name='denoised_input')
        
        x = denoised_input
        
        # ⭐ Deep hidden layers 추가!
        for i, dim in enumerate(self.cls_hidden_dims):
            x = layers.Dense(
                dim,
                kernel_initializer='glorot_uniform',
                name=f'cls_dense_{i+1}'
            )(x)
            
            x = layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-3,
                name=f'cls_bn_{i+1}'
            )(x)
            
            x = layers.Activation('relu', name=f'cls_relu_{i+1}')(x)
            
            # 앞쪽 레이어에만 dropout (과적합 방지)
            if i < len(self.cls_hidden_dims) - 1:
                x = layers.Dropout(0.3, name=f'cls_dropout_{i+1}')(x)
        
        # 최종 BAM 레이어
        cls_logits = BAMAssociativeLayer(
            input_dim=self.cls_hidden_dims[-1],  # 128
            output_dim=self.num_classes,  # 10
            steps=self.bam_steps,
            temp=self.bam_temp,
            name='bam_assoc'
        )(x)
        
        cls_output = layers.Activation('softmax', name='cls_output')(cls_logits)
        
        self.cls_model = keras.Model(
            inputs=denoised_input,
            outputs=cls_output,
            name='classification_bam_deep'
        )
    
    def compile_models(self, denoise_lr=1e-3, cls_lr=1e-3):
        """
        두 모델을 독립적으로 컴파일
        
        Args:
            denoise_lr: MF 학습률
            cls_lr: Classification 학습률 (높임!)
        """
        # Stage 1: MF 복원 모델
        self.denoise_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=denoise_lr,
                clipnorm=1.0,
                clipvalue=0.5
            ),
            loss='mae',
            metrics=[
                keras.metrics.MeanSquaredError(name="mse"),
                keras.metrics.MeanAbsoluteError(name="mae")
            ]
        )
        
        # Stage 2: Deep Classification 모델
        self.cls_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=cls_lr,  # 더 높은 learning rate
                clipnorm=1.0
            ),
            loss='categorical_crossentropy',
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
            ]
        )
        
        print("✓ Models compiled successfully (Deep Classification Network)")
    
    def train_stage1(self, x_noisy, x_clean, epochs=50, batch_size=128,
                     validation_split=0.2, callbacks=None):
        """
        Stage 1: MF 복원 학습 (기존과 동일)
        
        Args:
            x_noisy: 노이즈가 추가된 입력 (flattened)
            x_clean: 깨끗한 타겟 (flattened)
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
        
        history1 = self.denoise_model.fit(
            x_noisy, x_clean,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Stage 1 training complete!")
        return history1
    
    def train_stage2(self, x_noisy, y_labels, epochs=50, batch_size=128,
                     validation_split=0.2, callbacks=None):
        """
        Stage 2: Deep Classification 학습
        
        Args:
            x_noisy: 노이즈가 추가된 입력 (flattened)
            y_labels: 원-핫 인코딩된 레이블
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            callbacks: 콜백 리스트
            
        Returns:
            학습 히스토리
        """
        # 메모리 정리
        import gc
        gc.collect()
        
        print("\n" + "="*60)
        print("Stage 2: Training Deep Classification Network + BAM")
        print("="*60)
        print(f"Architecture: Deep NN + BAM")
        print(f"Hidden layers: {self.cls_hidden_dims}")
        print(f"BAM input dim: {self.cls_hidden_dims[-1]}")
        print(f"BAM output dim: {self.num_classes}")
        print(f"BAM steps: {self.bam_steps}")
        print(f"BAM temperature: {self.bam_temp}")
        
        # 1. MF로 훈련 데이터 복원 (배치 단위)
        print("\nGenerating denoised images for training...")
        
        n_samples = len(x_noisy)
        x_restored_list = []
        
        for i in range(0, n_samples, batch_size * 4):
            end_idx = min(i + batch_size * 4, n_samples)
            x_batch_restored = self.denoise_model.predict(
                x_noisy[i:end_idx],
                batch_size=batch_size,
                verbose=0
            )
            x_restored_list.append(x_batch_restored)
            
            if (i // (batch_size * 4)) % 10 == 0:
                print(f"  Processed {end_idx}/{n_samples} samples...")
        
        x_restored = np.concatenate(x_restored_list, axis=0)
        del x_restored_list
        
        print(f"✓ Restored images shape: {x_restored.shape}")
        
        # 2. Deep Classification Network 학습
        print("\nTraining Deep Classification Network with softmax + categorical crossentropy")
        history2 = self.cls_model.fit(
            x_restored, y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Stage 2 training complete!")
        return history2
    
    def predict(self, x_noisy, batch_size=128, verbose=0):
        """
        연쇄 추론: x_noisy → MF → x_restored → Deep NN + BAM → y_pred
        
        Args:
            x_noisy: 노이즈가 추가된 입력
            batch_size: 배치 크기
            verbose: 출력 레벨
            
        Returns:
            dict: {'denoised': 복원 이미지, 'predictions': 분류 결과}
        """
        # Stage 1: Denoising
        x_denoised = self.denoise_model.predict(x_noisy, batch_size=batch_size, verbose=verbose)
        
        # Stage 2: Classification
        y_pred = self.cls_model.predict(x_denoised, batch_size=batch_size, verbose=verbose)
        
        return {
            'denoised': x_denoised,
            'predictions': y_pred
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
        outputs = self.predict(x_noisy, batch_size=batch_size, verbose=1)
        
        # 복원 성능
        mse = np.mean((outputs['denoised'] - x_clean) ** 2)
        mae = np.mean(np.abs(outputs['denoised'] - x_clean))
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


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Testing SequentialBAMv2...")
    
    # 더미 데이터
    import numpy as np
    x_noisy = np.random.randn(100, 3072).astype('float32')
    x_clean = np.random.randn(100, 3072).astype('float32')
    y_labels = np.eye(10)[np.random.randint(0, 10, 100)]
    
    # 모델 생성
    model = SequentialBAMv2(
        input_dim=3072,
        denoise_latent=256,
        cls_hidden_dims=[1024, 512, 256, 128],  # Deep!
        num_classes=10
    )
    
    # 모델 컴파일
    model.compile_models(denoise_lr=1e-3, cls_lr=1e-3)
    
    print("\n[Denoise Model Summary]")
    model.denoise_model.summary()
    
    print("\n[Classification Model Summary]")
    model.cls_model.summary()
    
    print("\n✓ SequentialBAMv2 test passed")

