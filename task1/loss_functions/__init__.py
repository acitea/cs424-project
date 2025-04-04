from models.common import *
from piqa import SSIM

def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize a tensor normalized with mean and std.
    Args:
        tensor (torch.Tensor): Normalized tensor.
        mean (float): Mean used for normalization.
        std (float): Std used for normalization.
    Returns:
        torch.Tensor: Denormalized tensor.
    """
    return tensor * std + mean

class HingeAdversarialLoss(nn.Module):
    def forward(self, pred, is_real):
        if is_real:
            return F.relu(1 - pred).mean()
        else:
            return F.relu(1 + pred).mean()

class EdgeConsistencyLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.ssim = SSIM(n_channels=1, value_range=data_range)

        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[[[-1., 0., 1.], 
                                  [-2., 0., 2.], 
                                  [-1., 0., 1.]]]])
        sobel_y = torch.tensor([[[[-1., -2., -1.], 
                                  [0., 0., 0.], 
                                  [1., 2., 1.]]]])
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def rgb_to_grayscale(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def get_edge_map(self, x):
        x_gray = self.rgb_to_grayscale(x)
        
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Normalize edge magnitude to [0, 1]
        min_val = edge_mag.min().detach()
        max_val = edge_mag.max().detach()
        return (edge_mag - min_val) / (max_val - min_val + 1e-8)

    def forward(self, x, y):
        # Denormalize inputs from [-1, 1] back to [0, 1]
        x = denormalize(x, mean=0.5, std=0.5)
        y = denormalize(y, mean=0.5, std=0.5)

        # Compute edge maps
        edge_x = self.get_edge_map(x)
        edge_y = self.get_edge_map(y)

        # Compute SSIM between edge maps
        return 1.0 - self.ssim(edge_x, edge_y)

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False)
        self.sobel_y = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False)
        # Initialize Sobel kernels
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]*3]*3)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]*3]*3)
        self.sobel_x.weight = nn.Parameter(sobel_x.float())
        self.sobel_y.weight = nn.Parameter(sobel_y.float())
        
    def forward(self, generated, target):
        grad_gen = torch.sqrt(self.sobel_x(generated)**2 + self.sobel_y(generated)**2)
        grad_tar = torch.sqrt(self.sobel_x(target)**2 + self.sobel_y(target)**2)
        return F.l1_loss(grad_gen, grad_tar)

class IdentityPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim = EdgeConsistencyLoss(data_range=1.0)
        
    def forward(self, input, target):
        ssim_loss = self.ssim(input, target)
        gram_input = self.gram_matrix(input)
        gram_target = self.gram_matrix(target)
        style_loss = self.mse_loss(gram_input, gram_target)
        return 0.7 * ssim_loss + 0.3 * style_loss
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

class GradientPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = EdgeDetection()
        
    def forward(self, generated, target):
        grad_gen = self.sobel(generated)
        grad_target = self.sobel(target)
        return F.l1_loss(grad_gen, grad_target)

class EdgeGuidedLoss(nn.Module):
    def __init__(self, face_parser: nn.Module):
        super().__init__()
        self.edge_detector = EdgeDetection()
        self.parser = face_parser()
        
    def forward(self, x):
        # Generate edges
        grayscale = x.mean(dim=1, keepdim=True)
        edges = self.edge_detector(grayscale)
        
        # Generate pseudo labels
        with torch.no_grad():
            logits = self.parser(x)
            pseudo_labels = torch.argmax(logits, 1)
        
        # Calculate edge-aware loss
        edge_mask = (edges > 0.3).float()
        loss = F.cross_entropy(logits, pseudo_labels) * edge_mask
        return loss.mean()

class FaceStyleTransferLoss:
    def __init__(self, lambda_identity=10.0, lambda_facial=5.0, lambda_line=2.0):
        self.lambda_identity = lambda_identity
        self.lambda_facial = lambda_facial
        self.lambda_line = lambda_line
        
        # Feature extractor for identity (would be trained from scratch)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def identity_loss(self, x, y):
        # Extract identity features
        x_feat = self.feature_extractor(x)
        y_feat = self.feature_extractor(y)
        
        # Normalize features
        x_feat = F.normalize(x_feat, p=2, dim=1)
        y_feat = F.normalize(y_feat, p=2, dim=1)
        
        # Cosine similarity for identity preservation
        return 1 - F.cosine_similarity(x_feat, y_feat, dim=1).mean()
    
    def facial_component_loss(self, x, y, face_parsing):
        loss = 0
        for component in ['eyes', 'nose', 'mouth']:
            # Apply component mask
            mask = face_parsing[component]
            x_comp = x * mask
            y_comp = y * mask
            
            # Component-specific L1 loss
            loss += F.l1_loss(x_comp, y_comp)
        
        return loss
    
    def line_aware_loss(self, x, y):
        # Extract edges
        x_edges = self._extract_edges(x)
        y_edges = self._extract_edges(y)
        
        return F.l1_loss(x_edges, y_edges)
    
    def _extract_edges(self, img):
        # Simple edge detection using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        
        if img.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        edge_x = F.conv2d(img, sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(img, sobel_y, padding=1, groups=3)
        
        return torch.sqrt(edge_x**2 + edge_y**2)

class LineContinuityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilation = nn.MaxPool2d(3, stride=1, padding=1)
        self.erosion = -nn.MaxPool2d(3, stride=1, padding=1, return_indices=False)
        
    def morphological_grad(self, x):
        return self.dilation(x) + self.erosion(x)
    
    def forward(self, generated):
        grad = self.morphological_grad(generated)
        return torch.mean(torch.exp(-grad*10))

class PatchNCELoss(nn.Module):
    def __init__(self, tau=0.07):
        super(PatchNCELoss, self).__init__()
        self.tau = tau
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, f_q, f_k):
        # f_q and f_k are BxCxS tensors
        B, C, S = f_q.shape

        # Positive logit
        l_pos = torch.bmm(f_q.transpose(1, 2), f_k).view(B, S, 1)

        # Negative logits
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k.transpose(0, 2, 1))

        # Diagonal entries are not negatives
        identity_matrix = torch.eye(S, device=f_q.device).bool()
        l_neg.masked_fill_(identity_matrix.unsqueeze(0), float('-inf'))

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=2) / self.tau

        # Labels
        labels = torch.zeros(B * S, dtype=torch.long, device=f_q.device)

        # Flatten
        logits = logits.view(-1, S + 1)

        loss = self.cross_entropy_loss(logits, labels)
        return loss
