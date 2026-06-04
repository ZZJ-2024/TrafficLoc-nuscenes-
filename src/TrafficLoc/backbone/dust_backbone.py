import torch
import torch.nn as nn
from functools import partial
from itertools import repeat
import collections.abc
from dust3r.patch_embed import get_patch_embed

from croco.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from croco.models.blocks import Block

from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
import PIL.Image

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask
    
class DustEncoder(nn.Module):
    
    def __init__(self,
                 img_size=(512, 512),           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=1024,      # encoder feature dimension
                 enc_depth=24,           # encoder depth 
                 enc_num_heads=16,       # encoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',     # positional embedding (either cosine or RoPE100)
                 patch_embed_cls='PatchEmbedDust3R',
                 ):
        super(DustEncoder, self).__init__()
        
        self.patch_embed_cls = patch_embed_cls
        
        # patch embeddings  (with initialization done as in MAE)
        self.set_patch_embed(img_size, patch_size, enc_embed_dim)
        
        # mask generations
        self.set_mask_generator(self.patch_embed.num_patches, mask_ratio)
        
        self.pos_embed = pos_embed
        
        if pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)
        
        # transformer for the encoder
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # initializer weights
        self.initialize_weights()
        
    def set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        
    def set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
    
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # linears and layer norms
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _encode_image(self, image, true_shape, patch_embedding):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        if patch_embedding is not None:
            x = patch_embedding
            
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None
    
    def _encode_symmetrized(self, img1, patch_embedding):

        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        # warning! maybe the images have different portrait/landscape orientations

        feat1, pos1, _ = self._encode_image(img1, shape1, patch_embedding)
        
        return shape1, feat1, pos1
    
    def forward(self, image, patch_embedding):
        """_summary_

        Args:
            image (tensor): B x 3 x H x W
        """        
        batch_size , _, origin_h, origin_w = image.shape
        assert origin_h % 16 == 0 and origin_w % 16 == 0
        
        shape, feat, pos = self._encode_symmetrized(image, patch_embedding)
        
        feat = feat.reshape(batch_size, origin_h//16, origin_w//16, -1)
        
        feat=feat.contiguous()
        
        return feat
    
def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

if __name__ == "__main__":
    model = DustEncoder()
    print(model)
    
    # load image
    image_path = 'croco/assets/image_1871.jpg'
    square_ok = False
    size = 512
    img = exif_transpose(PIL.Image.open(image_path)).convert('RGB')
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    W2, H2 = img.size
    print(f' - adding {image_path} with resolution {W1}x{H1} --> {W2}x{H2}')
    input_image=ImgNorm(img)[None]  
    
    
    input_image = torch.randn((4,3,288,512))
    
    # load pretrain model
    model_path = "model_release/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    
    print(model.load_state_dict(ckpt['model'], strict=False))
    
    # test encoder forward process
    model = model.cuda()
    input_image = input_image.cuda()
    
    print(input_image.shape)
    output = model(input_image)
    print(output.shape)

    