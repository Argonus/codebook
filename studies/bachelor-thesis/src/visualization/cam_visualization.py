import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from typing import Dict, List

from src.utils.consts import TF_RECORD_DATASET, TF_BUFFER_SIZE, TF_BATCH_SIZE, DROP_NO_FINDING_CLASS, NO_FINDING_CLASS_IDX
from src.model.tensorflow_utils import load_dataset, optimize_dataset, filter_no_finding_class

class ModelCamVisualization:
    """
    Class for visualizing class activation maps (CAMs) for multiple models.
    """
    
    def __init__(self, models: Dict[str, tf.keras.Model], label_mappings_path: str, output_dir: str):
        """
        Initialize the model CAM visualization environment.
        :param models: Dictionary of model names and their corresponding models
        :param label_mappings_path: Path to the label mappings CSV file
        :param output_dir: Directory to save visualization results
        """
        self.models = models
        self.output_dir = output_dir
        self.has_no_finding_output = not DROP_NO_FINDING_CLASS
        self.no_finding_idx = NO_FINDING_CLASS_IDX
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = self._load_dataset()  
        self.label_names = self._load_label_mappings(label_mappings_path)
        self.class_examples = self._find_class_examples()

    def compare_models(self, class_name: str) -> None:
        """
        Compare CAMs for a specific class across all models.
        :param class_name: Name of the class to compare
        """
        class_idx = self.label_names.index(class_name)
        
        if class_idx not in self.class_examples:
            print(f"Warning: No example found for class '{class_name}'. Skipping visualization.")
            return
            
        img = self.class_examples[class_idx]
        display_img = img[0].numpy()
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
        
        axes[0].imshow(display_img)
        axes[0].set_title(f"Original: {class_name}")
        axes[0].axis('off')
        
        for i, (model_name, model) in enumerate(self.models.items(), start=1):
            heatmap = self._get_grad_cam(model, img, class_idx)
            
            cam_image = self._apply_heatmap(display_img, heatmap)
            
            axes[i].imshow(cam_image)
            axes[i].set_title(f"{model_name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Model Comparison for {class_name}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        output_file = os.path.join(self.output_dir, f"{class_name}_model_comparison.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"CAM model comparison for '{class_name}' saved to {output_file}")
        plt.close()

    def compare_classes(self, model_name: str) -> None:
        """
        Compare CAMs for different classes using a specific model.
        :param model_name: Name of the model to use for comparison
        """
        model = self.models[model_name]
        
        available_classes = sorted(list(self.class_examples.keys()))
        n_classes = len(available_classes)
        
        if n_classes == 0:
            print("No class examples found in the dataset. Skipping visualization.")
            return
            
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        for i, class_idx in enumerate(available_classes):
            img = self.class_examples[class_idx]
            class_name = self.label_names[class_idx]
            display_img = img[0].numpy()
            
            heatmap = self._get_grad_cam(model, img, class_idx)
            
            cam_image = self._apply_heatmap(display_img, heatmap)
            
            axes[i].imshow(cam_image)
            axes[i].set_title(f"{class_name}")
            axes[i].axis('off')
        
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.suptitle(f"Class Activation Maps for {model_name}", fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        output_file = os.path.join(self.output_dir, f"{model_name}_class_comparison.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"CAM class comparison for model '{model_name}' saved to {output_file}")
        plt.close()
        
    def _find_class_examples(self) -> Dict[int, np.ndarray]:
        """
        Find one example for each class in the dataset
        :return: Dictionary mapping class indices to example images
        """
        is_batched = False
        for x, y in self.dataset.take(1):
            if len(x.shape) == 4 and x.shape[0] > 1:
                is_batched = True
            break
        
        batched_dataset = self.dataset if is_batched else self.dataset.batch(128)
        
        class_examples = {}
        class_found = [False] * len(self.label_names)
        
        all_found = False
        
        for x_batch, y_batch in batched_dataset:
            if all_found:
                break
                
            for i in range(len(x_batch)):
                x = x_batch[i:i+1]
                y = y_batch[i]
                
                for class_idx, is_positive in enumerate(y.numpy()):
                    if is_positive > 0.5 and not class_found[class_idx]:
                        class_examples[class_idx] = x
                        class_found[class_idx] = True
                
                all_found = all(class_found)
                if all_found:
                    break
        
        missing_classes = [self.label_names[i] for i, found in enumerate(class_found) if not found]
        if missing_classes:
            print(f"Warning: Could not find examples for the following classes: {', '.join(missing_classes)}")
            
        return class_examples
        
    def _get_grad_cam(self, model: tf.keras.Model, img: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.
        
        :param model: TensorFlow model
        :param img: Preprocessed input image (4D tensor: 1 x height x width x channels)
        :param class_idx: Index of the target class
        :return: Heatmap as a numpy array
        """
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Concatenate):
                last_conv_layer_name = layer.name
                break
                
        if last_conv_layer_name is None:
            raise ValueError("Could not automatically find a convolutional layer in the model")
            
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img)
            class_channel = preds[:, class_idx]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
        
    def _apply_heatmap(self, img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Apply colorized heatmap on original image
        
        :param img: Original image (3D array: height x width x channels)
        :param heatmap: Heatmap from Grad-CAM
        :param alpha: Transparency factor
        :return: Superimposed visualization
        """
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed
        
    def _load_label_mappings(self, mappings_path: str) -> List[str]:
        df = pd.read_csv(mappings_path)
        df = df.sort_values('Index')

        if not self.has_no_finding_output:
            df = df[df['Index'] != self.no_finding_idx]

        return df['Label'].tolist()


    def _load_dataset(self) -> tf.data.Dataset:
        dataset = load_dataset(f"{TF_RECORD_DATASET}/test.tfrecord", TF_BUFFER_SIZE)
        if not self.has_no_finding_output:
            dataset = filter_no_finding_class(dataset)
        dataset = optimize_dataset(dataset, TF_BATCH_SIZE)
        
        return dataset