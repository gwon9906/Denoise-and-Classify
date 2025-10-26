"""
MTL BAM (Bidirectional Associative Memory) Model
Multi-Task Learning for simultaneous denoising and classification
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


class MTLBAM:
    """
    Multi-Task Learning BAM Model
    
    Simultaneously learns:
    1. Image reconstruction (denoising)
    2. Image classification
    
    Uses shared encoder with two decoder heads
    """
    
    def __init__(self, input_dim=3072, latent_dim=256, num_classes=10,
                 recon_weight=0.7, cls_weight=0.3, 
                 learning_rate=1e-3, recon_loss='mse'):
        """
        Args:
            input_dim: Input dimension (3072 for CIFAR-10)
            latent_dim: Latent dimension
            num_classes: Number of classes
            recon_weight: Weight for reconstruction loss
            cls_weight: Weight for classification loss
            learning_rate: Learning rate
            recon_loss: Reconstruction loss ('mse' or 'mae')
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight
        self.learning_rate = learning_rate
        self.recon_loss_type = recon_loss
        
        # Build model
        self.model = self._build_model()
        self._compile_model()
    
    def _build_model(self):
        """
        Build MTL BAM model
        """
        # Input
        inputs = layers.Input(shape=(self.input_dim,), name='noisy_input')
        
        # Shared Encoder
        enc1 = TiedDense(512, activation='relu', name='shared_enc1')
        enc2 = TiedDense(self.latent_dim, activation='relu', name='shared_enc2')
        
        x = enc1(inputs)
        x = layers.Dropout(0.2)(x)
        latent = enc2(x)
        latent = layers.Dropout(0.2)(latent)
        
        # Reconstruction Decoder (tied weights)
        dec2 = TiedDense(512, tied_to=enc2, activation='relu', name='recon_dec2')
        dec1 = TiedDense(self.input_dim, tied_to=enc1, activation='sigmoid', name='recon_dec1')
        
        x_recon = dec2(latent)
        x_recon = layers.Dropout(0.2)(x_recon)
        reconstruction = dec1(x_recon)
        
        # Classification Head
        x_cls = layers.Dense(256, activation='relu', name='cls_fc1')(latent)
        x_cls = layers.Dropout(0.3)(x_cls)
        x_cls = layers.Dense(128, activation='relu', name='cls_fc2')(x_cls)
        x_cls = layers.Dropout(0.3)(x_cls)
        classification = layers.Dense(self.num_classes, activation='softmax', name='cls_output')(x_cls)
        
        # Build model
        model = keras.Model(
            inputs=inputs,
            outputs={
                'reconstruction_output': reconstruction,
                'classification_output': classification
            },
            name='mtl_bam'
        )
        
        return model
    
    def _compile_model(self):
        """
        Compile model with weighted multi-task losses
        """
        # Loss functions
        if self.recon_loss_type == 'mse':
            recon_loss = 'mse'
        elif self.recon_loss_type == 'mae':
            recon_loss = 'mae'
        else:
            recon_loss = 'mse'
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'reconstruction_output': recon_loss,
                'classification_output': 'categorical_crossentropy'
            },
            loss_weights={
                'reconstruction_output': self.recon_weight,
                'classification_output': self.cls_weight
            },
            metrics={
                'reconstruction_output': ['mse', 'mae'],
                'classification_output': ['accuracy', 
                                          keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
            }
        )
        
        print("✓ MTL BAM model compiled")
        print(f"  Reconstruction weight: {self.recon_weight}")
        print(f"  Classification weight: {self.cls_weight}")
    
    def train(self, x_noisy, x_clean, y_labels, epochs=100, batch_size=128,
              validation_split=0.2, callbacks=None, verbose=1):
        """
        Train MTL model
        
        Args:
            x_noisy: Noisy input images (flattened)
            x_clean: Clean target images (flattened)
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
        print("Training MTL BAM")
        print("="*60)
        
        history = self.model.fit(
            x_noisy,
            {
                'reconstruction_output': x_clean,
                'classification_output': y_labels
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✓ Training complete!")
        return history
    
    def predict(self, x_noisy, batch_size=128):
        """
        Predict both reconstruction and classification
        
        Args:
            x_noisy: Noisy input images (flattened)
            batch_size: Batch size
        
        Returns:
            Dictionary with 'reconstruction' and 'classification' keys
        """
        predictions = self.model.predict(x_noisy, batch_size=batch_size, verbose=0)
        return predictions
    
    def evaluate_detailed(self, x_noisy, x_clean, y_labels, batch_size=128):
        """
        Detailed evaluation of both tasks
        
        Args:
            x_noisy: Noisy input images (flattened)
            x_clean: Clean reference images (flattened)
            y_labels: One-hot encoded labels
            batch_size: Batch size
        
        Returns:
            Dictionary with detailed metrics
        """
        print("\n" + "="*60)
        print("Evaluating MTL BAM")
        print("="*60)
        
        # Get predictions
        predictions = self.predict(x_noisy, batch_size=batch_size)
        x_recon = predictions['reconstruction_output']
        y_pred = predictions['classification_output']
        
        # Reconstruction metrics
        mse = np.mean((x_recon - x_clean) ** 2)
        mae = np.mean(np.abs(x_recon - x_clean))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Classification metrics
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_labels, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        
        # Top-3 accuracy
        top3_preds = np.argsort(y_pred, axis=1)[:, -3:]
        top3_acc = np.mean([true_classes[i] in top3_preds[i] for i in range(len(true_classes))])
        
        results = {
            'reconstruction': {
                'mse': float(mse),
                'mae': float(mae),
                'psnr': float(psnr)
            },
            'classification': {
                'accuracy': float(accuracy),
                'top3_accuracy': float(top3_acc)
            }
        }
        
        print("\nReconstruction Performance:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        
        print("\nClassification Performance:")
        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  Top-3 Acc:    {top3_acc:.4f}")
        
        return results
    
    def save_model(self, filepath):
        """
        Save model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """
        Load model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath, 
                                             custom_objects={'TiedDense': TiedDense})
        print(f"✓ Model loaded: {filepath}")


# Example usage
if __name__ == "__main__":
    print("MTL BAM Model for CIFAR-10")
    print("="*60)
    
    # Create model
    model = MTLBAM(
        input_dim=3072,      # 32*32*3
        latent_dim=256,
        num_classes=10,
        recon_weight=0.7,
        cls_weight=0.3
    )
    
    # Show architecture
    print("\n[Model Architecture]")
    model.model.summary()
    
    print("\n✓ MTL BAM created successfully!")