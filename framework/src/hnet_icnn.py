import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
# !/usr/bin/env python3
# Copyright 2020 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :hnets/mlp_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/14/2020
# @version        :1.0
# @python_version :3.6.10

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.ReLU(), dropout_rate=None):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None
        
        # Adjusting residual for matching shapes
        self.residual_adjust = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        
        if self.residual_adjust is not None:
            residual = self.residual_adjust(residual)
        
        out = out + residual
        return out

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)

class HyperDenseICNN(nn.Module):
    def __init__(self, target_shapes, convexify_id=None, convexify_fn=F.relu,
                 embedding_size=8, layers=[16, 16],
                 activation_fn=nn.ReLU(), dropout_rate=-1, verbose=True):
        nn.Module.__init__(self)
        ### Make constructor arguments internally available ###
        assert len(target_shapes) > 0
        self._layers = layers
        self._act_fn = activation_fn
        self._dropout_rate = dropout_rate
        self._target_shapes = target_shapes
        self._convex_id = convexify_id
        self._convex_fn = convexify_fn

        self._dropout = None
        if dropout_rate != -1:
            assert dropout_rate > 0 and dropout_rate < 1
            self._dropout = nn.Dropout(dropout_rate)

        ### Compute output size ###
        self.num_outputs = int(np.sum([np.prod(l)
                               for l in self._target_shapes]))

        ### Create fc layers ###
        fc_layers = nn.ModuleList()
        # self.residual_blocks = nn.ModuleList()
        layer_sizes = [embedding_size] + layers
        layer_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        for (in_features, out_features) in layer_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(self._act_fn)
            # self.residual_blocks.append(
            #     ResidualBlock(in_features, out_features, activation_fn, dropout_rate if self._dropout is not None else None)
            # )
            if self._dropout is not None:
                fc_layers.append(self._dropout)

        self.linear_layers = nn.Sequential(*fc_layers)
        
        self.output_layer = nn.Linear(layers[-1], self.num_outputs)
        # 初始化权重为小的值
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)
        # 初始化偏置为0
        nn.init.constant_(self.output_layer.bias, 0)

        self.apply(weights_init)

        if verbose:
            print('Hypernet for ICNN created.')
            print(self)

    def forward(self, embedding=None, convexify=True):
        # if embedding is None:
        #     embedding = self.placeholder
        h = self.linear_layers(embedding)
        # h = embedding
        # for block in self.residual_blocks:
        #     h = block(h)
        h = self.output_layer(h)
        h = h.mean(dim=0, keepdim=True)
        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h)
        # print(ret[9], flush=True)
        if convexify and self._convex_id is not None:
            for i in self._convex_id:
                ret[i] = self._convex_fn(ret[i])

        return ret

    def reset_parameters(self):
        self.output_layer.reset_parameters()

        for it in self.linear_layers:
            if isinstance(it, nn.Linear):
                it.reset_parameters()

    def _flat_to_ret_format(self, flat_out):
        """Helper function to convert flat hypernet output into desired output
        format.
        """
        assert len(flat_out.shape) == 2
        batch_size = flat_out.shape[0]

        ret = [[] for _ in range(batch_size)]
        ind = 0
        for s in self._target_shapes:
            num = int(np.prod(s))

            W = flat_out[:, ind:ind+num]
            W = W.view(batch_size, *s)

            for bind, W_b in enumerate(torch.split(W, 1, dim=0)):
                W_b = torch.squeeze(W_b, dim=0)
                assert np.all(np.equal(W_b.shape, s))
                ret[bind].append(W_b)

            ind += num

        if batch_size == 1:
            return ret[0]
        return ret


if __name__ == '__main__':
    pass
