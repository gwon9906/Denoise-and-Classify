# MTL 구조의 U-Net 모델 (conv 벡터 제거)
from tensorflow.keras import layers, Model, regularizers
import tensorflow as tf

def build_multitask_unet(input_shape_img, num_classes):
    """
    input_shape_img: (H, W, 3) - z-score noisy 이미지
    num_classes: 분류할 클래스 수
    """
    
    def conv_block(x, nf):
        """기본 컨볼루션 블록"""
        x = layers.Conv2D(nf, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(nf, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    # 입력 (이미지만)
    image_input = layers.Input(shape=input_shape_img, name="image_input")

    # Encoder (4 layers)
    c1 = conv_block(image_input, 32)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck (1 layer)
    b = conv_block(p4, 512)

    # Decoder (4 layers)
    d4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(b)
    d4 = layers.Concatenate()([d4, c4])
    d4 = conv_block(d4, 256)

    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d4)
    d3 = layers.Concatenate()([d3, c3])
    d3 = conv_block(d3, 128)

    d2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d3)
    d2 = layers.Concatenate()([d2, c2])
    d2 = conv_block(d2, 64)

    d1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d2)
    d1 = layers.Concatenate()([d1, c1])
    d1 = conv_block(d1, 32)

    # --- 복원 헤드: residual → 최종 복원 = noisy - residual ---
    residual_pred = layers.Conv2D(3, 1, activation='linear', name="residual_pred")(d1)
    restoration_head = layers.Subtract(name="restoration_output")([image_input, residual_pred])

    # --- 분류 헤드 ---
    flat = layers.GlobalAveragePooling2D()(b)
    dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(flat)
    dropout = layers.Dropout(0.5)(dense1)
    classification_head = layers.Dense(num_classes, activation='softmax', name="classification_output")(dropout)

    model = Model(inputs=image_input, outputs=[restoration_head, classification_head])
    return model


def create_model(input_shape=(32, 32, 3), num_classes=10):
    """
    모델 생성 함수
    
    Args:
        input_shape: 입력 이미지 형태 (H, W, C)
        num_classes: 분류할 클래스 수
    
    Returns:
        컴파일된 모델
    """
    model = build_multitask_unet(input_shape, num_classes)
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss={
            'restoration_output': 'mse',
            'classification_output': 'categorical_crossentropy'
        },
        loss_weights={
            'restoration_output': 1.0,
            'classification_output': 0.1
        },
        metrics={
            'restoration_output': ['mae'],
            'classification_output': ['accuracy']
        }
    )
    
    return model


if __name__ == "__main__":
    # 모델 생성 예제
    model = create_model(input_shape=(32, 32, 3), num_classes=10)
    model.summary()
