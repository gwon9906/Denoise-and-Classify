# BAM 모델 학습 스크립트
import tensorflow as tf
from tensorflow import keras
import numpy as np
from bam_model import create_bam_model, preprocess_cifar10_data


def train_bam_model(epochs=20, batch_size=128, noise_std=0.2, seed=42,
                   alpha=1.0, beta=0.5, learning_rate=1e-3, 
                   save_path=None, verbose=1):
    """
    BAM 모델 학습
    
    Args:
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        noise_std: 노이즈 표준편차
        seed: 랜덤 시드
        alpha: reconstruction loss weight
        beta: classification loss weight
        learning_rate: 학습률
        save_path: 모델 저장 경로
        verbose: 출력 레벨
    
    Returns:
        학습된 모델과 히스토리
    """
    # 시드 설정
    tf.keras.utils.set_random_seed(seed)
    
    # 데이터 전처리
    print("데이터 로딩 및 전처리 중...")
    data = preprocess_cifar10_data(noise_std=noise_std, seed=seed)
    
    # 모델 생성
    print("BAM 모델 생성 중...")
    model = create_bam_model(
        input_dim=3072, 
        latent_dim=128, 
        num_classes=data['num_classes'],
        alpha=alpha, 
        beta=beta, 
        learning_rate=learning_rate
    )
    
    # 학습
    print("BAM 모델 학습 시작...")
    history = model.fit(
        x=data['x_train_noisy'],
        y={"recon": data['x_train_clean'], "probs": data['y_train_1h']},
        validation_data=(data['x_test_noisy'], {"recon": data['x_test_clean'], "probs": data['y_test_1h']}),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    
    # 모델 저장
    if save_path:
        model.save(save_path)
        print(f"모델이 {save_path}에 저장되었습니다.")
    
    return model, history, data


def evaluate_model(model, data, num_samples=8):
    """
    모델 평가 및 시각화
    
    Args:
        model: 학습된 모델
        data: 테스트 데이터
        num_samples: 시각화할 샘플 수
    """
    print("BAM 모델 평가 중...")
    
    # 테스트 데이터로 예측
    y_pred = model.predict(data['x_test_noisy'][:num_samples], verbose=0)
    recons = y_pred["recon"].reshape(-1, 32, 32, 3)
    pred_labels = np.argmax(y_pred["probs"], axis=1)
    true_labels = data['y_test'][:num_samples].flatten()
    
    print(f"예측 라벨: {pred_labels}")
    print(f"실제 라벨: {true_labels}")
    print(f"정확도: {np.mean(pred_labels == true_labels):.3f}")
    
    return recons, pred_labels, true_labels


if __name__ == "__main__":
    # 하이퍼파라미터
    LATENT_DIM = 128
    ALPHA = 1.0   # reconstruction loss weight
    BETA = 0.5    # classification loss weight
    NOISE_STD = 0.2
    BATCH_SIZE = 128
    EPOCHS = 20
    SEED = 42
    
    # 모델 학습
    model, history, data = train_bam_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        noise_std=NOISE_STD,
        seed=SEED,
        alpha=ALPHA,
        beta=BETA,
        learning_rate=1e-3,
        save_path="best_bam_model.keras",
        verbose=1
    )
    
    # 모델 평가
    recons, pred_labels, true_labels = evaluate_model(model, data, num_samples=8)
