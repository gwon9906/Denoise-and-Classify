"""
Sequential BAM (Bidirectional Associative Memory) Model
Fixed version for CIFAR-10 denoising and classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TiedDense(layers.Layer):
    """
    Tied Dense layer - uses transposed weights for decoding
    """
    def __init__(self, units, tied_to=None, activation=None, **kwargs):
        super(TiedDense, self).__init__(**kwargs)
        self.units = units
        self.tied_to = tied_to
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        if self.tied_to is None:
            # Encoder: create new weights
            self.kernel = self.add_weight(
                name='kernel',
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True
            )
        else:
            # Decoder: use transposed encoder weights
            self.kernel = self.tied_to.kernel
            
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(TiedDense, self).build(input_shape)
    
    def call(self, inputs):
        if self.tied_to is None:
            # Encoder: normal multiplication
            output = tf.matmul(inputs, self.kernel)
        else:
            # Decoder: transposed multiplication
            output = tf.matmul(inputs, self.kernel, transpose_b=True)
        
        output = tf.nn.bias_add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_config(self):
        config = super(TiedDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config


class SequentialBAM:
    """
    Sequential BAM Model for Denoising and Classification
    
    Architecture:
    - Stage 1: Denoising with BAM (tied weights)
    - Stage 2: Classification on denoised images
    """
    
    def __init__(self, input_dim=3072, denoise_latent=256, cls_latent=128, num_classes=10):
        """
        Args:
            input_dim: Input dimension (32*32*3 = 3072 for CIFAR-10)
            denoise_latent: Latent dimension for denoising BAM
            cls_latent: Latent dimension for classification
            num_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.denoise_latent = denoise_latent
        self.cls_latent = cls_latent
        self.num_classes = num_classes
        
        # Build models
        self.denoise_model = self._build_denoise_model()
        self.cls_model = self._build_classification_model()
    
    def _build_denoise_model(self):
        """
        Build denoising BAM model with tied weights
        """
        # Input
        inputs = layers.Input(shape=(self.input_dim,), name='noisy_input')
        
        # Encoder layers (create these first to tie decoder to them)
        enc1 = TiedDense(512, activation='relu', name='denoise_enc1')
        enc2 = TiedDense(self.denoise_latent, activation='relu', name='denoise_enc2')
        
        # Encoder forward pass
        x = enc1(inputs)
        x = layers.Dropout(0.2)(x)
        latent = enc2(x)
        
        # Decoder layers (tied to encoder)
        dec2 = TiedDense(512, tied_to=enc2, activation='relu', name='denoise_dec2')
        dec1 = TiedDense(self.input_dim, tied_to=enc1, activation='sigmoid', name='denoise_dec1')
        
        # Decoder forward pass
        x = dec2(latent)
        x = layers.Dropout(0.2)(x)
        outputs = dec1(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='denoise_bam')
        return model
    
    def _build_classification_model(self):
        """
        Build classification model
        """
        # Input (will receive denoised images)
        inputs = layers.Input(shape=(self.input_dim,), name='denoised_input')
        
        # Hidden layers
        x = layers.Dense(512, activation='relu', name='cls_fc1')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.cls_latent, activation='relu', name='cls_fc2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='cls_output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='classification_bam')
        return model
    
    def compile_models(self, denoise_lr=1e-3, cls_lr=1e-3, denoise_loss='mse'):
        """
        Compile both models
        
        Args:
            denoise_lr: Learning rate for denoising model
            cls_lr: Learning rate for classification model
            denoise_loss: Loss function for denoising ('mse' or 'mae')
        """
        # Compile denoising model
        self.denoise_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=denoise_lr),
            loss=denoise_loss,
            metrics=['mse', 'mae']
        )
        
        # Compile classification model
        self.cls_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cls_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        print("✓ Models compiled successfully")
    
    def train_stage1(self, x_noisy, x_clean, epochs=100, batch_size=128, 
                     validation_split=0.2, callbacks=None, verbose=1):
        """
        Train denoising model (Stage 1)
        
        Args:
            x_noisy: Noisy input images (flattened)
            x_clean: Clean target images (flattened)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            callbacks: List of callbacks
            verbose: Verbosity mode
        
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("Stage 1: Training Denoising BAM")
        print("="*60)
        
        history = self.denoise_model.fit(
            x_noisy, x_clean,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✓ Stage 1 training complete!")
        return history
    
    def train_stage2(self, x_noisy, y_labels, epochs=100, batch_size=128,
                     validation_split=0.2, callbacks=None, verbose=1):
        """
        Train classification model (Stage 2)
        Uses denoised images from Stage 1
        
        Args:
            x_noisy: Noisy input images (flattened)
            y_labels: One-hot encoded labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            callbacks: List of callbacks
            verbose: Verbosity mode
        
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("Stage 2: Training Classification BAM")
        print("="*60)
        
        # Generate denoised images
        print("Generating denoised images for training...")
        x_denoised = self.denoise_model.predict(x_noisy, batch_size=batch_size, verbose=0)
        
        history = self.cls_model.fit(
            x_denoised, y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✓ Stage 2 training complete!")
        return history
    
    def predict(self, x_noisy, batch_size=128):
        """
        Full pipeline prediction: denoise + classify
        
        Args:
            x_noisy: Noisy input images (flattened)
            batch_size: Batch size for prediction
        
        Returns:
            Tuple of (denoised_images, class_predictions)
        """
        # Denoise
        x_denoised = self.denoise_model.predict(x_noisy, batch_size=batch_size, verbose=0)
        
        # Classify
        predictions = self.cls_model.predict(x_denoised, batch_size=batch_size, verbose=0)
        
        return x_denoised, predictions
    
    def evaluate(self, x_noisy, x_clean, y_labels, batch_size=128):
        """
        Evaluate both restoration and classification performance
        
        Args:
            x_noisy: Noisy input images (flattened)
            x_clean: Clean reference images (flattened)
            y_labels: One-hot encoded labels
            batch_size: Batch size
        
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("Evaluating Sequential BAM")
        print("="*60)
        
        # Get predictions
        x_denoised, predictions = self.predict(x_noisy, batch_size=batch_size)
        
        # Restoration metrics
        mse = np.mean((x_denoised - x_clean) ** 2)
        mae = np.mean(np.abs(x_denoised - x_clean))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Classification metrics
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_labels, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        
        # Top-3 accuracy
        top3_preds = np.argsort(predictions, axis=1)[:, -3:]
        top3_acc = np.mean([true_classes[i] in top3_preds[i] for i in range(len(true_classes))])
        
        results = {
            'restoration': {
                'mse': float(mse),
                'mae': float(mae),
                'psnr': float(psnr)
            },
            'classification': {
                'accuracy': float(accuracy),
                'top3_accuracy': float(top3_acc)
            }
        }
        
        print("\nRestoration Performance:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        
        print("\nClassification Performance:")
        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  Top-3 Acc:    {top3_acc:.4f}")
        
        return results
    
    def save_models(self, denoise_path, cls_path):
        """
        Save both models
        
        Args:
            denoise_path: Path to save denoising model
            cls_path: Path to save classification model
        """
        self.denoise_model.save(denoise_path)
        self.cls_model.save(cls_path)
        print(f"✓ Models saved:")
        print(f"  Denoise: {denoise_path}")
        print(f"  Classification: {cls_path}")
    
    def load_models(self, denoise_path, cls_path):
        """
        Load both models
        
        Args:
            denoise_path: Path to denoising model
            cls_path: Path to classification model
        """
        self.denoise_model = keras.models.load_model(denoise_path, 
                                                      custom_objects={'TiedDense': TiedDense})
        self.cls_model = keras.models.load_model(cls_path)
        print(f"✓ Models loaded:")
        print(f"  Denoise: {denoise_path}")
        print(f"  Classification: {cls_path}")


# Example usage
if __name__ == "__main__":
    print("Sequential BAM Model for CIFAR-10")
    print("="*60)
    
    # Create model
    model = SequentialBAM(
        input_dim=3072,      # 32*32*3
        denoise_latent=256,
        cls_latent=128,
        num_classes=10
    )
    
    # Show architecture
    print("\n[Denoising Model]")
    model.denoise_model.summary()
    
    print("\n[Classification Model]")
    model.cls_model.summary()
    
    print("\n✓ Sequential BAM created successfully!")