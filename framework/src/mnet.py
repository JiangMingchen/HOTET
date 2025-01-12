#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.batchnorm_layer import BatchNormLayer
from hypnettorch.utils.context_mod_layer import ContextModLayer
from hypnettorch.utils.torch_utils import init_params


class MainDenseICNN(nn.Module, MainNetInterface):
    def __init__(self, in_dim=1, hidden_layer_sizes=[32, 32, 32],
                 rank=1, activation='softplus', dropout=0.03,
                 strong_convexity=1e-6,
                 no_weights=False, init_weights=None,
                 out_fn=None, verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        # Tuple are not mutable.
        hidden_layer_sizes = list(hidden_layer_sizes)
        self.strong_convexity = strong_convexity
        self._hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout

        if activation == 'celu':
            self.activation = F.celu
        elif activation == 'softplus':
            self.activation = F.softplus
        elif self.activation == 'relu':
            activation = F.relu
        else:
            raise Exception('Activation is not specified or unknown.')

        self._no_weights = no_weights

        self._quadratic_layers = nn.ParameterList()
        self._convex_layers = nn.ParameterList()
        self._final_layer = nn.ParameterList()

        self._weights = nn.ParameterList(
            [self._quadratic_layers, self._convex_layers, self._final_layer])

        # Define and initialize linear weights.
        self._final_layer.append(nn.Parameter(
            torch.empty(1, hidden_layer_sizes[-1])
        ))

        in_features = in_dim
        for out_features in hidden_layer_sizes:
            tmp_quadratic_layer = nn.ParameterList()
            tmp_quadratic_layer.append(nn.Parameter(
                torch.empty(in_features, rank, out_features)
            ))
            tmp_quadratic_layer.append(nn.Parameter(
                torch.empty(out_features, in_features)
            ))
            tmp_quadratic_layer.append(nn.Parameter(torch.zeros(out_features)))

            # TODO: initialization
            # init_params(self._layer_weight_tensors[i])

            self._quadratic_layers.append(tmp_quadratic_layer)

        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        for (in_features, out_features) in sizes:
            # no bias
            self._convex_layers.append(nn.Parameter(
                torch.empty(out_features, in_features)
            ))

            # init_params(self._convex_layers[-1])

        # get the shapes of each layers
        self._param_shapes = [[*jt.shape]
                              for it in self._quadratic_layers for jt in it]
        #get the indices of convex layers
        self.convex_layer_indices = list(range(len(self._param_shapes), 
                                                len(self._param_shapes) + len(self._convex_layers)))
        
        self._param_shapes += [[*it.shape] for it in self._convex_layers]
        self._param_shapes += [[*(self._final_layer[0].shape)]]

        if no_weights:
            # self._is_properly_setup()
            return

        # initialization
        self.reset_parameters()

        # self._is_properly_setup()

    def forward(self, input, weights=None, distilled_params=None, condition=None, loaded_weights=False):
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # i.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.

        if weights is not None:
            quadratic_layers = []
            convex_layers = []
            final_layer = []

            depth = int(len(self.param_shapes) / 4)
            final_layer.append(weights[-1])
            for i in range(0, depth):
                quadratic_layers.append(weights[(3 * i):(3 * i + 3)])
                if i < (depth - 1):
                    convex_layers.append(weights[3 * depth + i])

            # self._quadratic_layers = nn.ParameterList(*quadratic_layers)
            # self._convex_layers = nn.ParameterList(*convex_layers)
            # self._final_layer = nn.ParameterList(*final_layer)
            # if loaded_weights:
            #     self._weights = nn.ParameterList([self._quadratic_layers, self._convex_layers, self._final_layer])
            # else:
            #     self.reset_parameters()
        else:
            quadratic_layers, convex_layers, final_layer = self._weights
        ###########################
        ### Forward Computation ###
        ###########################
        output = convexQuadratic(input, quadratic_layers[0])
        for i, (quadratic_weight, convex_weight) in enumerate(zip(quadratic_layers[1:], convex_layers)):
            quadratic_output = convexQuadratic(input, quadratic_weight)
            if self.droput != -1:
                quadratic_output = F.dropout(quadratic_output, self.droput)

            convex_output = F.linear(output, torch.clamp(convex_weight, min=0))
            if self.droput != -1:
                convex_output = F.dropout(convex_output, self.droput)

            output = quadratic_output + convex_output
            output = self.activation(output)
        
        #calculate 1/2 * x^2
        x_squared = input ** 2
        x_squared_half = 0.5 * x_squared.sum(dim=1, keepdim=True)
        
        output = F.linear(output, final_layer[0]) + .5 * self.strong_convexity * (
            input ** 2).sum(dim=1).reshape(-1, 1)
        return output + x_squared_half

    # # residual connection
    # def forward(self, input, weights=None, distilled_params=None, condition=None, loaded_weights=False):
    #     if self._no_weights and weights is None:
    #         raise Exception('Network was generated without weights. ' +
    #                         'Hence, "weights" option may not be None.')

    #     if weights is not None:
    #         quadratic_layers = []
    #         convex_layers = []
    #         final_layer = []

    #         depth = int(len(self.param_shapes) / 4)
    #         final_layer.append(weights[-1])
    #         for i in range(0, depth):
    #             quadratic_layers.append(weights[(3 * i):(3 * i + 3)])
    #             if i < (depth - 1):
    #                 convex_layers.append(weights[3 * depth + i])
    #     else:
    #         quadratic_layers, convex_layers, final_layer = self._weights

    #     # output = input  # Initialize output to input for residual connections
    #     output = convexQuadratic(input, quadratic_layers[0])
    #     for i, (quadratic_weight, convex_weight) in enumerate(zip(quadratic_layers[1:], convex_layers)):
    #         quadratic_output = convexQuadratic(input, quadratic_weight)
    #         if self.droput != -1:
    #             quadratic_output = F.dropout(quadratic_output, self.droput)

    #         convex_output = F.linear(output, torch.clamp(convex_weight, min=0))
    #         if self.droput != -1:
    #             convex_output = F.dropout(convex_output, self.droput)

    #         output = quadratic_output + convex_output  # Apply residual connection
    #         output = self.activation(output)

    #     final_output = F.linear(output, final_layer[0]) + .5 * self.strong_convexity * (
    #         input ** 2).sum(dim=1).reshape(-1, 1)
    #     final_output = input + final_output  # Final residual connection

    #     return final_output

    def distillation_targets(self):
        pass

    def push(self, input, weights=None):
        output = autograd.grad(
            outputs=self.forward(input, weights=weights), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output

    def push_nograd(self, input, weights=None):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        output.data = self.push(input, weights).data
        input.requires_grad_(False)
        return output

    def reset_parameters(self) -> None:
        for it in self._quadratic_layers:
            for jt in it:
                if len(jt.shape) > 1:
                    nn.init.kaiming_normal_(jt)

        for it in self._convex_layers:
            nn.init.kaiming_normal_(it)

        nn.init.kaiming_normal_(self._final_layer[0])

        self._weights = nn.ParameterList(
            [self._quadratic_layers, self._convex_layers, self._final_layer])


# def compute_stats(param_lists):

#     all_weights = torch.cat([p.view(-1) for sublist in param_lists for p in sublist.parameters()])
#     mean = all_weights.mean()
#     std = all_weights.std()
#     return mean, std

# def adjust_weights(source_param_list, target_mean, target_std):

#     for param in source_param_list:
#         adjusted_data = (param.data - param.data.mean()) / param.data.std() * target_std + target_mean
#         param.data.copy_(adjusted_data)

def compute_layer_stats(param_list, eps=1e-6):

    stats = []
    for param in param_list.parameters():
        mean = param.mean()
        std = param.std().clamp(min=eps)
        stats.append((mean, std))

    return stats


def adjust_layer_weights(tensor, layer_stats, eps=1e-6):

    for (target_mean, target_std) in layer_stats:
        adjusted_data = (tensor - tensor.mean()) / \
            tensor.std().clamp(min=eps) * target_std + target_mean
        tensor.data.copy_(adjusted_data)


def convexQuadratic(input, weights):
    '''Convex Quadratic Layer'''
    ###########################
    ### Forward Computation ###
    ###########################

    quad = input.matmul(weights[0].transpose(1, 0))
    quad = (quad.transpose(1, 0) ** 2).sum(dim=1)

    linear = F.linear(input, weights[1], weights[2])

    return quad + linear


if __name__ == '__main__':
    pass
