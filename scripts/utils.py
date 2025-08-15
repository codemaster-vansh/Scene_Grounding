"""
scripts/utils.py
"""
import json
from typing import Tuple
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

#Important Functions
def sine_2d_posenc(h:int,w:int,d_model:int,device:str):
    y,x = torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing="ij")
    d_half = d_model//2
    omega = torch.arange(d_half,device=device)/d_half
    omega = 1.0 / (10000**omega)

    y_enc = (y.flatten().unsqueeze(1) * omega)
    x_enc = (x.flatten().unsqueeze(1) * omega)

    pos = torch.cat(
        [torch.sin(y_enc),torch.cos(y_enc),torch.sin(x_enc),torch.cos(x_enc)],dim=1
    )

    return pos[:,:d_model]

def convert(boxes):
    cx,cy,w,h = boxes.unbind(-1)
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return torch.stack((x1,y1,x2,y2),dim=-1)

def generalized_iou(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_cxcywh : (B,N,4)
    tgt_cxcywh :  (B,M,4)
    Both in normalised (cx,cy,w,h) form.

    Returns GIoU (B,N,M)
    """

    b1 = convert(pred_cxcywh).unsqueeze(2)
    b2 = convert(tgt_cxcywh).unsqueeze(1)

    lt = torch.maximum(b1[..., :2], b2[..., :2])
    rb = torch.minimum(b1[..., 2:], b2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[...,0] * wh[...,1]

    area1 = (b1[..., 2]-b1[..., 0]) * (b1[..., 3]-b1[..., 1])
    area2 = (b2[..., 2]-b2[..., 0]) * (b2[..., 3]-b2[..., 1])
    union = area1 + area2 - intersection + eps

    iou = intersection/union

    lt_enc = torch.minimum(b1[..., :2], b2[..., :2])
    rb_enc = torch.maximum(b1[..., 2:], b2[..., 2:])
    wh_enc = rb_enc - lt_enc
    enc_area = wh_enc[..., 0] * wh_enc[..., 1] + eps

    giou = iou - (enc_area - union) / enc_area
    return giou

def box_loss(pred: torch.Tensor,tgt:  torch.Tensor) -> torch.Tensor:
    """
    pred : (B, Q, 4)   model outputs, (cx,cy,w,h)
    tgt  : (B, 4)      single GT box per image, (cx,cy,w,h)

    Returns scalar loss  =  mean_{B,Q}( L1  +  1-GIoU )
    """

    tgt_exp = tgt.unsqueeze(1)  # shape (B, 1, 4)

    l1 = F.l1_loss(pred, tgt_exp.expand_as(pred), reduction='none').sum(-1)
    giou = generalized_iou(pred, tgt_exp).squeeze(-1)

    return (l1 + (1.0 - giou)).mean()

def contrastive_loss(desc: torch.Tensor, img_tokens: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    CLS/first token of `desc` ↔ mean-pooled image tokens.
    Batch-shift provides negatives.
    """
    B = desc.size(0)
    assert B > 1, "Contrastive loss needs batch_size > 1 for negatives"

    # positive embeddings
    phrase_emb = desc[:, 0, :]            # (B, 256)
    img_emb    = img_tokens.mean(1)       # (B, 256)

    # cosine similarities
    pos_sim = F.cosine_similarity(phrase_emb, img_emb, dim=-1)          # (B,)

    # negatives: roll images one step
    img_emb_neg = torch.roll(img_emb, shifts=1, dims=0)                 # (B,256)
    neg_sim = F.cosine_similarity(phrase_emb, img_emb_neg, dim=-1)      # (B,)

    loss = F.relu(neg_sim - pos_sim + margin).mean()
    return loss


def visualize_prediction(model, image_path, phrase, tokenizer, device, img_size=224):
    model.eval()

    # 1. Load & preprocess image
    img = Image.open(image_path).convert("RGB")
    # If using torchvision transforms
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = tf(img).unsqueeze(0).to(device)  # (1,3,H,W)

    # 2. Tokenize the phrase
    tokens = tokenizer(phrase,
                       padding='max_length',
                       truncation=True,
                       max_length=20,
                       return_tensors="pt")
    tokens = {k: v.to(device) for k,v in tokens.items()}

    # 3. Forward pass through model
    with torch.no_grad():
        boxes_pred, _, _ = model(img_tensor, tokens)  # boxes_pred shape = (1, Q, 4)

    boxes_pred = boxes_pred[0].cpu()  # remove batch dim, (Q,4)

    # 4. Plot image with predicted boxes
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img)

    # Boxes are (cx,cy,w,h) normalized to [0,1]
    # Convert to (x1,y1,w,h) in image pixel coordinates
    W, H = img.size
    for i, box in enumerate(boxes_pred):
        cx, cy, w, h = box
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        width = w * W
        height = h * H

        # Draw rectangle on image
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor='r',
            facecolor='none',
            label=f'Box {i}'
        )
        ax.add_patch(rect)

    ax.set_title("Predicted Boxes")
    plt.axis("off")
    plt.show()


#CLASS DEFINITIONS

# 1) Dataset
class RefCOCODataset(Dataset):
    """
    Calls the processed RefCOCO dataset we made
    """
    def __init__(self,DATA_DIR, split:str,tokenizer=None,img_size: tuple =(224,224)):
        split_dir = DATA_DIR / split
        ann_path = DATA_DIR/f"anns_{split}.json"
        with open(ann_path) as f: self.anns = json.load(f)

        self.img_dir = split_dir
        self.tokenizer = tokenizer
        mean,std = (0.485,0.456,0.406),(0.229,0.224,0.225)
        self.tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    def __len__(self): return len(self.anns)

    def __getitem__(self, index):
        ann = self.anns[index]
        img = Image.open(self.img_dir/f"image_{ann['image_id']:04}.jpg").convert("RGB")
        W,H = img.size

        #Normalize
        x1,y1,x2,y2 = ann['bbox']
        w = x2 - x1
        h = y2 - y1
        cx = x1 + (0.5)*w
        cy = y1 + (0.5)*h
        bbox = torch.tensor([cx/W,cy/H,w/W,h/H],dtype=torch.float32)

        img = self.tf(img)
        tokens = self.tokenizer(
            ann['phrase'],padding='max_length',truncation=True,max_length=20,return_tensors="pt"
        )
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return {"image":img,"bbox":bbox,"tokens":tokens,"phrase":ann["phrase"]}
    
#2) ResNet backbone
class ResNetBackbone(nn.Module):
    """
    Calls the ResNet_50 Model
    """
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(*list(model.children())[:-2])
        self.proj = nn.Conv2d(2048,256,1)

        for p in self.stem[:6].parameters():
            p.requires_grad_ = False

    def forward(self,x) -> Tuple[torch.Tensor,int,int]:
        feat = self.proj(self.stem(x))
        feat = feat.float()
        B,C,H,W = feat.shape

        return feat.flatten(2).transpose(1,2),H,W
    
#3) Cross Modal Fusion Encoder

class CrossModalFusionEncoder(nn.Module):
    """
    Main Encoder Layer, just an wrapper of DualStreamFusion
    """
    def __init__(self,layers=2,dim=256,heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([DualStreamFusionLayer(dim,heads) for _ in range(layers)])

    def forward(self,img_tokens,txt_tokens):
        for blk in self.blocks:
            img_tokens,txt_tokens = blk(img_tokens,txt_tokens)
        return img_tokens, txt_tokens
    
class DualStreamFusionLayer(nn.Module):
    """
    One block exactly in the order:
    1) Self Attention (Text and Image Embeddings Separately)
    2) Image -> Text (Cross Attention)
    3) Text -> Image (Cross Attention)
    4) FFN (Separate)
    """

    def __init__(self,dim=256,heads=4,drop=0.1):
        super().__init__()
        self.self_img = nn.MultiheadAttention(dim,heads,dropout=drop,batch_first=True)
        self.self_txt = nn.MultiheadAttention(dim,heads,dropout=drop,batch_first=True)

        self.i2t = nn.MultiheadAttention(dim,heads,dropout=drop,batch_first=True)
        self.t2i = nn.MultiheadAttention(dim,heads,dropout=drop,batch_first=True)

        self.norm_i1 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.norm_i2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_i3 = nn.LayerNorm(dim)
        self.norm_t3 = nn.LayerNorm(dim)

        self.ffn_img = nn.Sequential(
            nn.Linear(dim,4*dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4*dim,dim)
        )

        self.ffn_text = nn.Sequential(
            nn.Linear(dim,4*dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4*dim,dim)
        )

    def forward(self,img,text):
        _img, _ = self.self_img(img,img,img)
        img = self.norm_i1(img + _img)

        _txt, _ = self.self_txt(text,text,text)
        text = self.norm_t1(text + _txt)

        _img, _ = self.t2i(img,text,text)
        img = self.norm_i2(img + _img)

        _text, _ = self.i2t(text,img,img)
        text = self.norm_t2(text + _text)

        img = self.norm_i3(img + self.ffn_img(img))
        text = self.norm_t3(text + self.ffn_text(text))

        return img, text
    
#4) Main Model
class FinalModel(nn.Module):
    """
    Wraps all classes. Final Model. Forward hook Function represents the model flow
    """
    def __init__(self,textbackbone,d_model=256,n_heads=4,ff_dim=1024,num_dec_layers=2):
        super().__init__()
        #encoders
        self.backbone = ResNetBackbone()
        self.text_enc = textbackbone
        self.text_projection = nn.Linear(768,d_model)

        #fusion
        self.fusion = CrossModalFusionEncoder()

        #decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=num_dec_layers
        )

        #bbox head
        self.bbox_convert = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,4),
            nn.Sigmoid()
        )

    def forward(self,batch_imgs,tokens):
        B = batch_imgs.size()

        #Step 1 - Visual token processing
        v_tokens, H, W = self.backbone(batch_imgs)   # (B,Ni,256)
        pos = sine_2d_posenc(H,W,v_tokens.size(-1),batch_imgs.device)
        v_tokens = v_tokens + pos.unsqueeze(0)

        #Step 2 - Textual token processing
        token_dict = {k:v.to(batch_imgs.device) for k,v in tokens.items()}
        t_feat = self.text_enc(**token_dict).last_hidden_state
        t_tokens = self.text_projection(t_feat)

        #cross modal fusion
        fused_img_embs,fused_txt_embs = self.fusion(v_tokens,t_tokens)

        #decoder
        dec_out = self.decoder(tgt=fused_txt_embs,memory=fused_img_embs)

        #predictions
        boxes = self.bbox_convert(dec_out)
        return boxes,dec_out, fused_img_embs
    
