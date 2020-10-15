import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 n_head=8,
                 num_encoder_layer=6,
                 num_decoder_layer=6,
                 dim_feedforward=2048,
                 drop_out=0.1,
                 normalize_before=False,
                 return_intermediate_dec=False):
        super(Transformer, self).__init__()
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(d_model,
                                          n_head,
                                          dim_feedforward,
                                          drop_out,
                                          normalize_before,
                                          num_encoder_layer, encoder_norm)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(d_model,
                                          n_head,
                                          dim_feedforward,
                                          drop_out,
                                          normalize_before,
                                          num_decoder_layer,
                                          decoder_norm,
                                          return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # [h x w, bs, c]
        # hs [num_layers,h x w,bs, c]

        # return [num_layers,bs, hxw, c]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 normalize_before=False,
                 num_layers=6,
                 norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, normalize_before
        ) for _ in range(num_layers)])
        self.norm = nn.Identity() if norm is None else norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

    @staticmethod
    def add_pos_embed(x, pos=None):
        return x if pos is None else x + pos

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.add_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.add_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask=None,
                src_key_padding_mask=None,
                pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model,
                 n_head=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 normalize_before=False,
                 num_layer=6, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(
            d_model, n_head, dim_feedforward, dropout, normalize_before
        ) for _ in range(num_layer)])
        self.norm = nn.Identity() if norm is None else norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        output = tgt
        intermediate = list()
        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return self.norm(output).unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head=6, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act = nn.ReLU(inplace=True)

        self.normalize_before = normalize_before

    @staticmethod
    def add_pos_embed(x, pos=None):
        return x if pos is None else x + pos

    def forward_pre(self, tgt, memory,
                    tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.add_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.add_pos_embed(tgt2, query_pos),
                                   key=self.add_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward_post(self, tgt, memory,
                     tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.add_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.add_pos_embed(tgt, query_pos),
                                   key=self.add_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


if __name__ == '__main__':
    input_tensor = torch.rand(size=(16, 4, 512))
    input_mask = torch.rand(size=(4, 16)) > 0.5
    net = TransformerEncoder(d_model=512, n_head=8)
    out = net(input_tensor, src_key_padding_mask=input_mask)
    print(out.shape)
