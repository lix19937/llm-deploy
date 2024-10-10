
## transformer 模块优化的核心   

+ ffn (gemm)   

+ mha (softmax gemm)  

+ ca   

+ ln  (pre norm /post norm)    
self.operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    ```py
    for layer in self.operation_order:
        # temporal self attention
        if layer == 'self_attn':
            identity = query # !!!
            query = self.attentions[attn_index](
                query, identity if self.pre_norm else None, inner_add=False)
            attn_index += 1
    
        elif layer == 'norm':
            query = self.norms[norm_index](query + identity) # post norm 
            norm_index += 1
    
        # spaital cross attention
        elif layer == 'cross_attn':
            identity = query # !!!
            query = self.attentions[attn_index](
                query, identity if self.pre_norm else None, inner_add=False)
            attn_index += 1
    
        # ffn 
        elif layer == 'ffn':
            identity = query # !!!
            query = self.ffns[ffn_index](
                query, identity if self.pre_norm else None, inner_add=False)
            ffn_index += 1
    
    return query
    ```



