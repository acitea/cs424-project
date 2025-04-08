from PIL import Image
import torch
import time
import os
import logging
import numpy as np
from bg_removal import remove_background
from generators.task2.PokemonGenerator import PokemonGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseGANModel:
    """Base class for GAN models"""
    def __init__(self, name):
        self.name = name
    
    def generate(self, input_path, output_path):
        """
        Generate an image transformation
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError("Subclasses must implement the generate method")


class StyleTransferGAN(BaseGANModel):
    """Style Transfer GAN implementation"""
    def __init__(self):
        super().__init__("style_transfer")
        logger.info(f"Initializing {self.name} model...")
    
    def generate(self, input_path, output_path):
        """Apply style transfer to the input image"""
        try:
            # Simulate processing time
            time.sleep(2)
            
            # Load image
            img = Image.open(input_path)
            
            # Mock style transfer with a sepia filter
            width, height = img.size
            pixels = img.load()
            for i in range(width):
                for j in range(height):
                    r, g, b = pixels[i, j][:3]
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    pixels[i, j] = (min(tr, 255), min(tg, 255), min(tb, 255))
                    
            # Save the result
            img.save(output_path)
            return True
        except Exception as e:
            logger.error(f"Error in style transfer: {e}")
            return False


class PokemonGAN(BaseGANModel):
    """Pokemon style GAN using pretrained model"""
    def __init__(self, reverse: bool = False):
        super().__init__("pokemon_generator")
        logger.info(f"Initializing {self.name} model...")
        self.residual_blocks = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = self._load_model(reverse)
    
    def generate(self, input_path, output_path):
        """Transform the input image into Pokemon style"""
        try:
            logger.info(f"Processing image with Pokemon GAN: {input_path}")
            
            # First remove background for cleaner results
            temp_path = input_path + ".tmp.png"
            remove_background(input_path, temp_path)
            
            # Load the image and preprocess
            img = Image.open(temp_path).convert("RGB")
            img = img.resize((96, 96))

            # convert to tensor and normalize to [-1, 1]
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Generate the transformed image
            with torch.no_grad():
                output_tensor = self.model(img_tensor)

            # convert output tensor back to image
            output_array = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            # denormalize from [-1, 1] to [0, 255]
            output_array = ((output_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            output_image = Image.fromarray(output_array)
            output_image.save(output_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return True
        except Exception as e:
            logger.error(f"Error in Pokemon GAN generation: {e}")
            return False
    
    def _load_model(self, reverse: bool = False):
        """Load the pretrained Pokemon generator model"""
        try:
            # Create model instance
            generator = PokemonGenerator(in_channels=3, num_residual_blocks=self.residual_blocks, use_attention=True)
            
            # Get the correct path to the weights file
            
            weights_path = os.path.join("weights", "task2", "forward", "PokeMalCycleGAN-96x96-G_AB-475-best.pth") if not reverse else os.path.join("weights", "task2","reverse", "PokeMalCycleGAN-96x96-G_BA-475-best.pth")
            logger.info(f"Loading model weights from: {weights_path}")
            
            # Load the checkpoint
            # If using CPU-only system, we need to map the tensor locations
            if self.device.type == 'cuda':
                checkpoint = torch.load(weights_path)
            else:
                checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
            
            # Some checkpoints might store the state dict directly or under a key like 'model'
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load the state dict into the model
            generator.load_state_dict(state_dict)
            generator.to(self.device)
            generator.eval()
            
            logger.info("Pokemon generator model loaded successfully")
            return generator
            
        except Exception as e:
            logger.error(f"Error loading Pokemon generator model: {e}")
            raise

# Factory function to get the appropriate model
def get_model(task_id, reverse: bool = False):
    """
    Get the appropriate GAN model based on task ID
    
    Args:
        task_id (str): Task identifier
        
    Returns:
        BaseGANModel: The appropriate GAN model instance
    """
    models = {
        "task1": StyleTransferGAN(),
        "task2": PokemonGAN(reverse=reverse),
    }
    
    return models.get(task_id)