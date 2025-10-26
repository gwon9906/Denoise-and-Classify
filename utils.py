# 업데이트된 공통 유틸리티 함수들
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
import json
import pickle

def setup_gpu_memory(memory_limit_mb=8192):
    """
    GPU 메모리 설정 (8GB 제한)
    
    Args:
        memory_limit_mb: 메모리 제한 (MB)
    """
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 메모리 제한 설정
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            print(f"✓ GPU memory growth enabled and limited to {memory_limit_mb}MB")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            return False
    else:
        print("⚠ No GPU found, using CPU")
        return False


def add_noise_by_snr(clean_data, snr_db, noise_type='gaussian'):
    """
    특정 SNR로 노이즈 추가
    
    Args:
        clean_data: 깨끗한 데이터 (N, H, W, C) 또는 (N, D)
        snr_db: 목표 SNR (dB)
        noise_type: 'gaussian', 'sp', 'burst'
    
    Returns:
        노이즈가 추가된 데이터
    """
    signal_power = np.mean(clean_data ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std, clean_data.shape).astype('float32')
        noisy_data = clean_data + noise
    
    elif noise_type == 'sp':
        # Salt & Pepper noise
        noisy_data = clean_data.copy()
        noise_ratio = min(0.5, noise_std * 2)
        
        # Salt (1.0)
        salt_mask = np.random.random(clean_data.shape) < noise_ratio / 2
        noisy_data[salt_mask] = 1.0
        
        # Pepper (0.0)
        pepper_mask = np.random.random(clean_data.shape) < noise_ratio / 2
        noisy_data[pepper_mask] = 0.0
    
    elif noise_type == 'burst':
        burst_types = ['dead_pixels', 'column_row', 'block']
        burst_type = np.random.choice(burst_types)
        noisy_data = add_burst_noise(clean_data, noise_std, burst_type)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return np.clip(noisy_data, 0.0, 1.0).astype('float32')


def add_burst_noise(clean_data, noise_std, burst_type='dead_pixels'):
    """
    Burst 노이즈 추가
    
    Args:
        clean_data: 깨끗한 데이터 (N, H, W, C)
        noise_std: 노이즈 강도
        burst_type: 'dead_pixels', 'column_row', 'block'
    """
    noisy_data = clean_data.copy()
    N, H, W, C = clean_data.shape
    
    affected_ratio = min(0.3, noise_std * 3)
    
    if burst_type == 'dead_pixels':
        for i in range(N):
            num_dead = int(H * W * affected_ratio)
            dead_positions = np.random.choice(H * W, num_dead, replace=False)
            
            for pos in dead_positions:
                row = pos // W
                col = pos % W
                noisy_data[i, row, col, :] = np.random.choice([0.0, 1.0])
    
    elif burst_type == 'column_row':
        for i in range(N):
            num_lines = max(1, int(max(H, W) * affected_ratio / 10))
            
            for _ in range(num_lines):
                if np.random.random() < 0.5:
                    col = np.random.randint(0, W)
                    noisy_data[i, :, col, :] = np.random.choice([0.0, 1.0])
                else:
                    row = np.random.randint(0, H)
                    noisy_data[i, row, :, :] = np.random.choice([0.0, 1.0])
    
    elif burst_type == 'block':
        for i in range(N):
            num_blocks = max(1, int(10 * affected_ratio))
            
            for _ in range(num_blocks):
                block_h = np.random.randint(2, max(3, H // 4))
                block_w = np.random.randint(2, max(3, W // 4))
                
                start_h = np.random.randint(0, H - block_h + 1)
                start_w = np.random.randint(0, W - block_w + 1)
                
                block_value = np.random.choice([0.0, 1.0])
                noisy_data[i, start_h:start_h+block_h, start_w:start_w+block_w, :] = block_value
    
    return noisy_data


def get_callbacks(model_name, monitor='val_loss', patience=30, initial_lr=1e-3, epochs=200):
    """
    학습 콜백 생성
    
    Args:
        model_name: 모델 이름
        monitor: 모니터링할 메트릭
        patience: Early stopping patience
        initial_lr: 초기 학습률
        epochs: 전체 에폭 수
    """
    final_lr = initial_lr * 0.1
    decay_rate = (final_lr / initial_lr) ** (1 / epochs)
    
    callbacks = []
    
    # 1. Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 2. Model Checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f'weights/{model_name}_best.keras',
        monitor=monitor,
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 3. Learning Rate Scheduler (Exponential Decay)
    def lr_schedule(epoch, lr):
        return initial_lr * (decay_rate ** epoch)
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
    callbacks.append(lr_scheduler)
    
    # 4. CSV Logger
    csv_logger = keras.callbacks.CSVLogger(
        f'logs/{model_name}_training.csv',
        append=False
    )
    callbacks.append(csv_logger)
    
    # 5. TensorBoard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=f'logs/{model_name}',
        histogram_freq=0,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    return callbacks


def clear_memory():
    """
    메모리 정리
    """
    print("\n" + "="*60)
    print("Clearing memory...")
    print("="*60)
    
    keras.backend.clear_session()
    gc.collect()
    
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.keras.backend.clear_session()
            print("✓ GPU memory cleared")
        except:
            print("⚠ Could not clear GPU memory explicitly")
    
    print("✓ Memory cleared\n")


def save_history(history, model_name):
    """
    학습 히스토리 저장
    """
    os.makedirs('history', exist_ok=True)
    with open(f'history/{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✓ History saved: history/{model_name}_history.pkl")


def save_results(results, model_name):
    """
    평가 결과 저장
    """
    os.makedirs('results', exist_ok=True)
    with open(f'results/{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: results/{model_name}_results.json")


def load_history(model_name):
    """
    학습 히스토리 로드
    """
    try:
        with open(f'history/{model_name}_history.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None


def load_results(model_name):
    """
    평가 결과 로드
    """
    try:
        with open(f'results/{model_name}_results.json', 'r') as f:
            return json.load(f)
    except:
        return None


def create_directories():
    """
    필요한 디렉토리 생성
    """
    directories = ['data', 'weights', 'history', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created:", ", ".join(directories))


def print_training_config(epochs, batch_size, validation_split, initial_lr):
    """
    학습 설정 출력
    """
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Final LR: {initial_lr * 0.1} (10%)")


def calculate_psnr(img1, img2):
    """
    PSNR 계산
    
    Args:
        img1: 첫 번째 이미지
        img2: 두 번째 이미지
    
    Returns:
        PSNR 값 (dB)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
