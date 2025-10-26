# CIFAR-10 Denoising & Classification - MTL vs Sequential Models

## 📋 프로젝트 개요

초저 SNR 환경(-30~-10dB)에서 노이즈 제거 및 분류를 수행하는 6개의 딥러닝 모델 비교 연구

### 비교 모델

#### Sequential Models (2-stage)
1. **Sequential BAM**: Dense layers + BAM 양방향 연상
2. **Sequential CAE**: Convolutional layers (Skip connection ❌)
3. **Sequential U-Net**: Convolutional layers + Skip connection ✅

#### MTL Models (1-stage, End-to-End)
4. **MTL BAM**: Dense layers + BAM 양방향 연상
5. **MTL CAE**: Convolutional layers (Skip connection ❌)
6. **MTL U-Net**: Convolutional layers + Skip connection ✅

---

## 🎯 실험 설계

### 데이터셋
- **Base**: CIFAR-10 (50,000 train / 10,000 test)
- **증강 후**: 150,000 train / 30,000 test

#### 노이즈 구성 (균등 분포)
- **3가지 노이즈 타입** (각 50,000장):
  - Gaussian noise
  - Salt & Pepper noise
  - Burst noise (3종류: Dead Pixels, Column/Row Defects, Block Bursts)

- **5가지 SNR 레벨** (각 10,000장):
  - -30 dB
  - -25 dB
  - -20 dB
  - -15 dB
  - -10 dB

### 학습 설정
```python
Epochs: 200
Batch size: 128
Validation split: 20% (120K train / 30K val)
Early stopping: patience=30
Initial LR: 1e-3
Final LR: 1e-4 (10%, Exponential decay)
Loss: MAE (restoration), Categorical CrossEntropy (classification)
```

### Loss Weights (MTL 모델)
- **BAM & CAE**: recon_weight=0.7, cls_weight=0.3
- **U-Net**: recon_weight=0.6, cls_weight=0.4

---

## 📁 프로젝트 구조

```
.
├── data/                           # 증강된 데이터셋
│   ├── x_train_augmented.npy       # 노이즈 이미지 (150K)
│   ├── y_train_augmented.npy       # 레이블
│   ├── x_train_clean.npy           # 원본 이미지 (복원 타겟)
│   ├── x_test_augmented.npy        # 테스트 노이즈 (30K)
│   ├── y_test_augmented.npy
│   ├── x_test_clean.npy
│   ├── train_noise_info.csv        # 노이즈 메타정보
│   └── test_noise_info.csv
│
├── models/                         # 모델 정의 파일들
│   ├── bam_sequential.py
│   ├── bam_mtl.py
│   ├── cae_sequential.py
│   ├── cae_mtl.py
│   ├── unet_sequential.py
│   └── unet_mtl.py
│
├── weights/                        # 학습된 모델 가중치
│   ├── sequential_bam_denoise.keras
│   ├── sequential_bam_classification.keras
│   ├── sequential_cae_restore.keras
│   ├── sequential_cae_classification.keras
│   ├── sequential_unet_restore.keras
│   ├── sequential_unet_classification.keras
│   ├── mtl_bam.keras
│   ├── mtl_cae.keras
│   └── mtl_unet.keras
│
├── history/                        # 학습 히스토리
│   ├── sequential_bam_stage1_history.pkl
│   ├── sequential_bam_stage2_history.pkl
│   ├── mtl_bam_history.pkl
│   └── ...
│
├── results/                        # 평가 결과
│   ├── sequential_bam_results.json
│   ├── mtl_bam_results.json
│   ├── comprehensive_results.csv
│   ├── noise_type_results.json      # NEW: 노이즈 타입별 결과
│   ├── snr_level_results.json       # NEW: SNR 레벨별 결과
│   ├── noise_type_summary.csv       # NEW: 노이즈 요약
│   ├── snr_level_summary.csv        # NEW: SNR 요약
│   ├── key_findings.txt             # NEW: 핵심 발견사항
│   ├── performance_comparison.png
│   ├── training_curves_mtl.png
│   ├── noise_type_analysis.png      # NEW: 노이즈 분석 차트
│   └── snr_level_analysis.png       # NEW: SNR 분석 차트
│
├── logs/                           # 학습 로그 (TensorBoard + CSV)
│   ├── sequential_bam_stage1_training.csv
│   ├── mtl_bam/
│   └── ...
│
├── data_preprocessing.ipynb        # 데이터 증강
├── train_all_models_part1.ipynb   # 학습 Part 1 (설정)
├── train_all_models_part2.ipynb   # 학습 Part 2 (Sequential)
├── train_all_models_part3.ipynb   # 학습 Part 3 (MTL)
├── train_all_models_part4.ipynb   # 학습 Part 4 (전체 비교)
├── train_all_models_part5.ipynb   # 학습 Part 5 (상세 분석)
├── utils.py                        # 유틸리티 함수
└── README.md                       # 이 파일
```

---

## 🚀 실행 방법

### 1. 데이터 전처리
```bash
jupyter notebook data_preprocessing.ipynb
```
- CIFAR-10 데이터를 150,000장으로 증강
- 3가지 노이즈 × 5가지 SNR 레벨
- `data/` 폴더에 저장

### 2. 모델 학습 (순차 실행)
```bash
jupyter notebook train_all_models_part1.ipynb  # 환경 설정
jupyter notebook train_all_models_part2.ipynb  # Sequential 모델
jupyter notebook train_all_models_part3.ipynb  # MTL 모델
jupyter notebook train_all_models_part4.ipynb  # 전체 결과 비교
jupyter notebook train_all_models_part5.ipynb  # 상세 분석 (노이즈/SNR별)
```

각 Part가 완료되면 자동으로 메모리를 정리하고 다음 Part로 진행

---

## 💻 시스템 요구사항

### 하드웨어
- **CPU**: Intel i7-12700K (또는 동급)
- **GPU**: NVIDIA RTX 3070 Ti 8GB
- **RAM**: 16GB 이상 권장
- **Storage**: 20GB 이상 (데이터 + 모델 가중치)

### 소프트웨어
```
Python >= 3.8
TensorFlow >= 2.10
NumPy
Pandas
Matplotlib
Seaborn
tqdm
```

### 설치
```bash
pip install tensorflow numpy pandas matplotlib seaborn tqdm
```

---

## 📊 평가 메트릭

### Restoration (복원)
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **PSNR** (Peak Signal-to-Noise Ratio, dB)

### Classification (분류)
- **Accuracy**
- **Top-3 Accuracy**

---

## 🔬 실험 결과 예시

결과는 `results/comprehensive_results.csv`에 저장됩니다:

| Model | Type | Architecture | Recon MSE | Recon PSNR (dB) | Classification Acc |
|-------|------|-------------|-----------|-----------------|-------------------|
| sequential_bam | Sequential | BAM | 0.0234 | 16.31 | 0.7234 |
| sequential_cae | Sequential | CAE | 0.0198 | 17.03 | 0.7556 |
| sequential_unet | Sequential | UNET | 0.0176 | 17.54 | 0.7823 |
| mtl_bam | MTL | BAM | 0.0245 | 16.11 | 0.7345 |
| mtl_cae | MTL | CAE | 0.0211 | 16.76 | 0.7612 |
| mtl_unet | MTL | UNET | 0.0189 | 17.24 | 0.7734 |

---

## 📈 시각화

### 1. 성능 비교 차트 (Part 4)
`results/performance_comparison.png`
- Reconstruction MSE
- Reconstruction PSNR
- Classification Accuracy
- PSNR vs Accuracy Trade-off (Scatter plot)

### 2. 학습 곡선 (Part 4)
`results/training_curves_mtl.png`
- Total Loss
- Reconstruction Loss
- Classification Loss
- Reconstruction MAE
- Classification Accuracy
- Learning Rate Schedule

### 3. 노이즈 타입별 분석 (Part 5) 🆕
`results/noise_type_analysis.png`
- PSNR by Noise Type (Line & Bar charts)
- MSE by Noise Type
- Classification Accuracy by Noise Type
- PSNR Heatmap (Model × Noise Type)

### 4. SNR 레벨별 분석 (Part 5) 🆕
`results/snr_level_analysis.png`
- PSNR vs Input SNR (Line chart)
- MSE vs Input SNR
- Classification Accuracy vs Input SNR
- PSNR Improvement over SNR
- PSNR Heatmap (Model × SNR Level)

---

## 🎓 연구 포인트

### 비교 분석
1. **Sequential vs MTL**: 2-stage vs End-to-End 학습
2. **Architecture**: Dense(BAM) vs Conv(CAE) vs Conv+Skip(U-Net)
3. **Special Mechanism**: BAM 양방향 연상의 효과
4. **Noise Robustness**: 노이즈 타입별 성능 차이 🆕
5. **SNR Sensitivity**: SNR 레벨에 따른 성능 변화 🆕

### 상세 분석 항목 (Part 5) 🆕
- **By Noise Type**: 각 노이즈(Gaussian, S&P, Burst)별 모델 성능
- **By SNR Level**: 각 SNR(-30~-10dB)별 모델 성능
- **Sequential vs MTL**: 조건별 상세 비교
- **Statistical Analysis**: 평균, 표준편차, 개선도
- **Key Findings**: 핵심 발견사항 자동 추출

### 기대 결과
- **U-Net**: Skip connection으로 최고의 복원 성능 예상
- **MTL**: End-to-End 학습으로 분류 성능 우수 예상
- **BAM**: 양방향 연상 메커니즘의 효과 검증
- **Noise-specific**: Burst 노이즈에서 모델 간 성능 차이 클 것으로 예상 🆕
- **SNR-dependent**: 높은 SNR일수록 MTL의 이점 증가 예상 🆕

---

## ⚙️ 커스터마이징

### Noise Configuration
`data_preprocessing.ipynb`에서 수정:
```python
snr_levels = [-30, -25, -20, -15, -10]  # SNR 레벨
noise_types = ['gaussian', 'sp', 'burst']  # 노이즈 타입
```

### Training Hyperparameters
`train_all_models_part1.ipynb`에서 수정:
```python
EPOCHS = 200
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
INITIAL_LR = 1e-3
```

### Loss Weights
각 모델 생성 시:
```python
# MTL 모델
mtl_bam = MTLBAM(recon_weight=0.7, cls_weight=0.3)
mtl_unet = UNetMTL(recon_weight=0.6, cls_weight=0.4)
```

---

## 🐛 문제 해결

### GPU 메모리 부족
```python
# utils.py 또는 Part1에서 메모리 제한 조정
setup_gpu_memory(memory_limit_mb=6144)  # 8GB → 6GB
```

### 학습 중 커널 크래시
- Batch size 감소: `BATCH_SIZE = 64`
- Early stopping patience 감소: `patience=20`
- 각 Part 실행 후 커널 재시작

---

## 📝 논문 작성 시 참고사항

### 실험 재현성
- Random seed 고정: `np.random.seed(42)`
- 데이터 split 고정: 동일한 증강 데이터 사용
- 모든 하이퍼파라미터 기록

### 통계적 유의성
- 여러 번 실험 후 평균/표준편차 계산
- T-test 또는 Wilcoxon test 수행

---

## 👥 기여자

- 프로젝트 작성자: [Your Name]
- 연구 지도: [Advisor Name]

---

## 📄 라이선스

MIT License

---

## 📧 문의

질문이나 문제가 있으시면 이슈를 등록해주세요.

---

## 🙏 감사의 말

- TensorFlow/Keras 팀
- CIFAR-10 데이터셋 제공자
- Anthropic Claude (코드 작성 지원)

---

**Happy Training! 🚀**
