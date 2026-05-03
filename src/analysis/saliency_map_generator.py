        import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


class SaliencyMapGenerator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook for Grad-CAM
        self.gradients = {}
        self.activations = {}
        self.hooks = []  # Keep track of hooks for cleanup
        
    def register_hooks(self):
        """Register hooks to capture gradients and activations for Grad-CAM"""
        # Clear existing hooks first
        self.clear_hooks()
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on the last convolutional layer
        target_layer = None
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
        
        if target_layer is not None:
            hook1 = target_layer.register_forward_hook(save_activation('last_conv'))
            hook2 = target_layer.register_backward_hook(save_gradient('last_conv'))
            self.hooks.extend([hook1, hook2])
    
    def clear_hooks(self):
        """Clear all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = {}
        self.activations = {}
    
    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.clear_hooks()
        
    def register_hooks(self):
        """Register hooks to capture gradients and activations for Grad-CAM"""
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on the last convolutional layer
        # Assuming the last conv layer is in features[-3] (before final ReLU and MaxPool)
        target_layer = None
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
        
        if target_layer is not None:
            target_layer.register_forward_hook(save_activation('last_conv'))
            # Use register_backward_hook to avoid deprecation warning
            target_layer.register_backward_hook(save_gradient('last_conv'))
    
    def vanilla_gradient_saliency(self, image, target_class=None):
        """
        Generate vanilla gradient saliency map
        
        Args:
            image: Input image tensor (C, H, W)
            target_class: Target class index. If None, uses predicted class
        
        Returns:
            saliency_map: Numpy array of saliency values
            prediction: Model's prediction
        """
        # Prepare input
        input_tensor = image.unsqueeze(0).to(self.device)  # Add batch dimension
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        
        # Use target class or predicted class
        if target_class is None:
            target_class = prediction
        
        # Backward pass
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients
        if input_tensor.grad is None:
            print("Warning: No gradients computed. Returning zeros.")
            return np.zeros(image.shape[-2:]), prediction
            
        gradients = input_tensor.grad.data.squeeze(0)  # Remove batch dimension
        
        # Calculate saliency map (take maximum across color channels)
        saliency_map = torch.max(torch.abs(gradients), dim=0)[0]
        saliency_map = saliency_map.detach().cpu().numpy()
        
        return saliency_map, prediction
    
    def gradcam_saliency(self, image, target_class=None):
        """
        Generate Grad-CAM saliency map
        
        Args:
            image: Input image tensor (C, H, W)
            target_class: Target class index. If None, uses predicted class
        
        Returns:
            gradcam_map: Numpy array of Grad-CAM values
            prediction: Model's prediction
        """
        try:
            # Register hooks
            self.register_hooks()
            
            # Prepare input
            input_tensor = image.unsqueeze(0).to(self.device)
            input_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
            # Use target class or predicted class
            if target_class is None:
                target_class = prediction
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
            
            # Check if gradients and activations are available
            if 'last_conv' not in self.gradients or 'last_conv' not in self.activations:
                print("Warning: Could not capture gradients or activations. Using vanilla gradient instead.")
                self.clear_hooks()
                return self.vanilla_gradient_saliency(image, target_class)
            
            # Get gradients and activations
            gradients = self.gradients['last_conv']
            activations = self.activations['last_conv']
            
            # Calculate weights (global average pooling of gradients)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Calculate Grad-CAM
            gradcam = torch.sum(weights * activations, dim=1).squeeze(0)
            gradcam = F.relu(gradcam)  # Apply ReLU to focus on positive influence
            
            # Resize to input image size
            original_size = image.shape[-2:]
            gradcam = F.interpolate(gradcam.unsqueeze(0).unsqueeze(0), 
                                   size=original_size, mode='bilinear', align_corners=False)
            gradcam = gradcam.squeeze().detach().cpu().numpy()
            
            # Normalize
            if gradcam.max() > 0:
                gradcam = gradcam / gradcam.max()
            
            return gradcam, prediction
            
        except Exception as e:
            print(f"Error in Grad-CAM generation: {e}")
            print("Falling back to vanilla gradient saliency...")
            return self.vanilla_gradient_saliency(image, target_class)
        
        finally:
            # Always clear hooks
            self.clear_hooks()
    
    def visualize_saliency(self, original_image, saliency_map, method_name, 
                          prediction=None, true_label=None, save_path=None):
        """
        Visualize saliency map overlaid on original image
        
        Args:
            original_image: Original image as PIL Image or numpy array
            saliency_map: Saliency map as numpy array
            method_name: Name of the saliency method
            prediction: Model's prediction
            true_label: Ground truth label
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert original image to numpy if needed
        if isinstance(original_image, torch.Tensor):
            if original_image.dim() == 3:  # C, H, W
                original_image = original_image.permute(1, 2, 0)
            original_image = original_image.cpu().numpy()
        elif isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure values are in [0, 1] range
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Saliency map
        im1 = axes[1].imshow(saliency_map, cmap='hot', alpha=0.8)
        axes[1].set_title(f'{method_name} Saliency Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(saliency_map, cmap='hot', alpha=0.4)
        title = f'{method_name} Overlay'
        if prediction is not None:
            title += f'\nPredicted: {prediction}'
        if true_label is not None:
            title += f' | True: {true_label}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()

def generate_saliency_for_dataset(model, dataset, device, num_samples=5, 
                                output_dir='saliency_maps', class_names=None):
    """
    Generate saliency maps for a sample of images from the dataset
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device to run on
        num_samples: Number of samples to generate saliency maps for
        output_dir: Directory to save visualizations
        class_names: List of class names for better visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize saliency generator
    saliency_gen = SaliencyMapGenerator(model, device)
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]
        
        # Get original image (before normalization)
        # Re-create the original image from the dataset
        img_path = os.path.join(dataset.image_dir, dataset.image_files[idx])
        original_pil = Image.open(img_path).convert("RGB")
        if hasattr(dataset, 'target_size') and dataset.target_size:
            original_pil = original_pil.resize(dataset.target_size)
        
        print(f"\nProcessing image {i+1}/{len(indices)}: {dataset.image_files[idx]}")
        
        # Generate vanilla gradient saliency
        vanilla_saliency, prediction = saliency_gen.vanilla_gradient_saliency(image)
        
        # Generate Grad-CAM saliency
        gradcam_saliency, _ = saliency_gen.gradcam_saliency(image)
        
        # Get class names if provided
        pred_name = class_names[prediction] if class_names else prediction
        true_name = class_names[true_label] if class_names else true_label
        
        # Visualize vanilla gradient
        save_path_vanilla = os.path.join(output_dir, f'vanilla_gradient_{i+1}.png')
        saliency_gen.visualize_saliency(
            original_pil, vanilla_saliency, 'Vanilla Gradient',
            prediction=pred_name, true_label=true_name, save_path=save_path_vanilla
        )
        
        # Visualize Grad-CAM
        save_path_gradcam = os.path.join(output_dir, f'gradcam_{i+1}.png')
        saliency_gen.visualize_saliency(
            original_pil, gradcam_saliency, 'Grad-CAM',
            prediction=pred_name, true_label=true_name, save_path=save_path_gradcam
        )


# Quick test function
def test_single_image(model, image_path, device, target_size=(100, 100)):
    """
    Test saliency generation on a single image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    
    # Convert to tensor
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
    
    # Generate saliency maps
    saliency_gen = SaliencyMapGenerator(model, device)
    
    vanilla_saliency, prediction = saliency_gen.vanilla_gradient_saliency(image_tensor)
    gradcam_saliency, _ = saliency_gen.gradcam_saliency(image_tensor)
    
    # Visualize
    saliency_gen.visualize_saliency(image, vanilla_saliency, 'Vanilla Gradient', prediction=prediction)
    saliency_gen.visualize_saliency(image, gradcam_saliency, 'Grad-CAM', prediction=prediction)
    
    return vanilla_saliency, gradcam_saliency, prediction
