

import torch
from loguru import logger 

# Each module is a dummy implementation

class NORM(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.act = torch.nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True) # --------

  def forward(self, x): # torch.Size([1, 16000, 256])
      out = self.act(x)
      return out
  

class FFN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.act = torch.nn.ReLU() # --------

  def forward(self, x, identity=None, inner_add=True):
      out = self.act(x)
      if identity is None:
          identity = x
      if inner_add:    
        return identity + out
      return out

class SelfAttention(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.act = torch.nn.GELU() # --------

  def forward(self, x, identity=None, inner_add=True):
      out = self.act(x)
      if identity is None:
          identity = x
      if inner_add:    
        return identity + out
      return out
  

class CrossAttention(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.act = torch.nn.Sigmoid() # --------

  def forward(self, x, identity=None, inner_add=True):
      out = self.act(x)
      if identity is None:
          identity = x
      if inner_add:    
        return identity + out
      return out
  

class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.pre_norm = False
        self.operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        self.attentions = [SelfAttention(), CrossAttention()]
        self.ffns = [FFN()]
        self.norms = [NORM(), NORM(), NORM()]

    def forward(self, query):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query, identity if self.pre_norm else None)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, identity if self.pre_norm else None)
                attn_index += 1
                identity = query

            # ffn 
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
   

    def forward_eq(self, query):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = None

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                identity = query
                query = self.attentions[attn_index](
                    query, identity if self.pre_norm else None, inner_add=False)
                attn_index += 1

            elif layer == 'norm':
                query = self.norms[norm_index](query + identity)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                identity = query
                query = self.attentions[attn_index](
                    query, identity if self.pre_norm else None, inner_add=False)
                attn_index += 1

            # ffn 
            elif layer == 'ffn':
                identity = query
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None, inner_add=False)
                ffn_index += 1

        return query

logger.info("start ...")

model = TransformerBlock()
x = torch.randn(1, 16000, 256, dtype=torch.float32)
t = model.forward(x)
r = model.forward_eq(x)

ret = torch.equal(t, r)
logger.info(ret)
