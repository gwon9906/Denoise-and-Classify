# MF-BAM MTL 및 Cascade 모델
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from .bam_mf import MF, build_mf_module, build_decoder
from .bam import BAM, BAMAssociativeLayer

# ============================================
# MTL 구조 (MF features → BAM)
# ============================================

def build_mf_bam_mtl(input_shape=(3072,), num_classes=10):
    """
    MF-BAM MTL 모델
    
    구조:
    Input → MF Encoder → [Decoder → Restoration, BAM → Classification]
    
    Args:
        input_shape: 입력 shape (flatten)
        num_classes: 분류 클래스 수
        
    Returns:
        Keras Model with 2 outputs: [restoration, classification]
    """
    inp = layers.Input(shape=input_shape, name='flatten_input')
    
    # MF 인코더 (특성 추출)
    mf_feat = build_mf_module(inp, hidden_units=[1024, 768, 512, 256])
    
    # 복원 헤드
    rec = build_decoder(mf_feat, decoder_units=[512, 768, 1024], original_dim=input_shape[0])
    
    # 분류 헤드 (BAM)
    cls_logits = BAMAssociativeLayer(
        mf_feat.shape[-1],
        num_classes,
        steps=5,
        temp=0.5,
        name='bam_assoc'
    )(mf_feat)
    
    cls = layers.Activation('softmax', name='classification_output')(cls_logits)
    
    return Model(inputs=inp, outputs=[rec, cls], name="MF_BAM_MTL")


# ============================================
# Cascade 구조 (MF → Restoration → BAM)
# ============================================

def build_mf_bam_cascade(input_shape=(3072,), num_classes=10):
    """
    MF-BAM Cascade 모델
    
    구조:
    Input → MF → Restored Image → BAM → Classification
    
    Args:
        input_shape: 입력 shape (flatten)
        num_classes: 분류 클래스 수
        
    Returns:
        Keras Model with 2 outputs: [restoration, classification]
    """
    inp = layers.Input(shape=input_shape, name='flatten_input')
    
    # MF 인코더
    mf_feat = build_mf_module(inp, hidden_units=[1024, 768, 512, 256])
    
    # 복원
    rec = build_decoder(mf_feat, decoder_units=[512, 768, 1024], original_dim=input_shape[0])
    
    # 복원된 이미지를 BAM 입력으로 사용
    cls_logits = BAMAssociativeLayer(
        rec.shape[-1],  # restored image dimension
        num_classes,
        steps=5,
        temp=0.5,
        name='bam_assoc'
    )(rec)
    
    cls = layers.Activation('softmax', name='classification_output')(cls_logits)
    
    return Model(inputs=inp, outputs=[rec, cls], name="MF_BAM_CASCADE")


# ============================================
# MTLModel 클래스 (기존 호환성 유지)
# ============================================

class MTLModel(tf.keras.Model):
    """
    MF-BAM MTL 모델 (복원+분류 동시 출력)
    
    기존 코드와의 호환성을 위해 유지
    """
    
    def __init__(self, mf_module: MF = None, bam_module: BAM = None,
                 input_dim=3072, num_classes=10):
        """
        Args:
            mf_module: MF 모듈 (선택적)
            bam_module: BAM 모듈 (선택적)
            input_dim: 입력 차원
            num_classes: 분류 클래스 수
        """
        super().__init__()
        
        # MF 모듈
        if mf_module is None:
            self.mf = MF(input_dim=input_dim, encoder_dims=[1024, 768, 512, 256])
        else:
            self.mf = mf_module
        
        # BAM 모듈
        if bam_module is None:
            latent_dim = 256  # MF의 마지막 차원
            self.bam = BAM(input_dim=latent_dim, output_dim=num_classes)
        else:
            self.bam = bam_module
    
    def call(self, x, training=None):
        # MF 모듈로 입력 복원 및 은닉표현 얻기
        recon, z = self.mf(x, training=training)
        
        # 은닉표현을 BAM 분류기에 통과시켜 클래스 패턴 출력
        class_out = self.bam(z)
        
        # Softmax 적용
        class_out = tf.nn.softmax(class_out)
        
        # 두 출력을 튜플로 반환
        return recon, class_out


# ============================================
# 모델 생성 함수
# ============================================

def create_mf_bam_model(model_type='mtl',
                        input_shape=(3072,),
                        num_classes=10,
                        recon_weight=0.7,
                        cls_weight=0.3,
                        learning_rate=2e-4):
    """
    MF-BAM 모델 생성 및 컴파일
    
    Args:
        model_type: 'mtl' 또는 'cascade'
        input_shape: 입력 shape
        num_classes: 분류 클래스 수
        recon_weight: 복원 손실 가중치
        cls_weight: 분류 손실 가중치
        learning_rate: 학습률
        
    Returns:
        컴파일된 Keras Model
    """
    if model_type == 'mtl':
        model = build_mf_bam_mtl(input_shape, num_classes)
    elif model_type == 'cascade':
        model = build_mf_bam_cascade(input_shape, num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "restoration_output": keras.losses.MeanSquaredError(),
            "classification_output": keras.losses.CategoricalCrossentropy()
        },
        loss_weights={
            "restoration_output": recon_weight,
            "classification_output": cls_weight
        },
        metrics={
            "restoration_output": [keras.metrics.MeanAbsoluteError(name='mae')],
            "classification_output": [
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
            ]
        }
    )
    
    return model


# ============================================
# 사용 예제
# ============================================

if __name__ == "__main__":
    print("Creating MF-BAM models...")
    
    # MTL 모델 생성
    print("\n" + "="*60)
    print("MTL Model")
    print("="*60)
    mtl_model = create_mf_bam_model(
        model_type='mtl',
        input_shape=(3072,),
        num_classes=10
    )
    mtl_model.summary()
    
    # Cascade 모델 생성
    print("\n" + "="*60)
    print("Cascade Model")
    print("="*60)
    cascade_model = create_mf_bam_model(
        model_type='cascade',
        input_shape=(3072,),
        num_classes=10
    )
    cascade_model.summary()
    
    print("\n✓ Models created successfully")
