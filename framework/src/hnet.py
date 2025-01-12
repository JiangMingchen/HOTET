#!/usr/bin/env python3
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

from collections import defaultdict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils import init_utils as iutils


class HyperDenseICNN(nn.Module, HyperNetInterface):
    def __init__(self, target_shapes, uncond_in_size=0, cond_in_size=8,
                 layers=(100, 100), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=2, dropout_rate=-1, use_batch_norm=False):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        assert len(target_shapes) > 0
        if cond_in_size == 0 and num_cond_embs > 0:
            warn('Requested that conditional weights are managed, but ' +
                 'conditional input size is zero! Setting "num_cond_embs" to ' +
                 'zero.')
            num_cond_embs = 0
        elif not no_cond_weights and num_cond_embs == 0 and cond_in_size > 0:
            warn('Requested that conditional weights are internally ' +
                 'maintained, but "num_cond_embs" is zero.')
        # Do we maintain conditional weights internally?
        has_int_cond_weights = not no_cond_weights and num_cond_embs > 0
        # Do we expect external conditional weights?
        has_ext_cond_weights = no_cond_weights and num_cond_embs > 0

        ### Make constructor arguments internally available ###
        self._uncond_in_size = uncond_in_size
        self._cond_in_size = cond_in_size
        self._layers = layers
        self._act_fn = activation_fn
        self._no_uncond_weights = no_uncond_weights
        self._no_cond_weights = no_cond_weights
        self._num_cond_embs = num_cond_embs
        self._dropout_rate = dropout_rate
        self._use_spectral_norm = False
        self._use_batch_norm = use_batch_norm

        ### Setup attributes required by interface ###
        self._target_shapes = target_shapes
        self._num_known_conds = self._num_cond_embs
        self._unconditional_param_shapes_ref = []

        self._has_bias = use_bias
        self._has_fc_out = True
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_uncond_weights and has_int_cond_weights else nn.ParameterList()
        self._hyper_shapes_learned = None if not no_uncond_weights and has_ext_cond_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._dropout = None
        if dropout_rate != -1:
            assert dropout_rate > 0 and dropout_rate < 1
            self._dropout = nn.Dropout(dropout_rate)

        ### Create conditional weights ###
        for _ in range(num_cond_embs):
            assert cond_in_size > 0
            if not no_cond_weights:
                self._internal_params.append(nn.Parameter(
                    data=torch.Tensor(cond_in_size), requires_grad=True))
                torch.nn.init.normal_(
                    self._internal_params[-1], mean=0., std=1.)
            else:
                self._hyper_shapes_learned.append([cond_in_size])
                self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append([cond_in_size])
            # Embeddings belong to the input, so we just assign them all to
            # "layer" 0.
            self._param_shapes_meta.append({
                'name': 'embedding',
                'index': -1 if no_cond_weights else
                len(self._internal_params)-1,
                'layer': 0
            })
        # print(len(self._internal_params))
        ### Create batch-norm layers ###
        # We just use even numbers starting from 2 as layer indices for
        # batchnorm layers.
        if use_batch_norm:
            self._add_batchnorm_layers(layers, no_uncond_weights,
                                       bn_layers=list(
                                           range(2, 2*len(layers)+1, 2)),
                                       distill_bn_stats=False, bn_track_stats=True)

        ### Create fully-connected hidden-layers ###
        in_size = uncond_in_size + cond_in_size
        if len(layers) > 0:
            # We use odd numbers starting at 1 as layer indices for hidden
            # layers.
            self._add_fc_layers([in_size, *layers[:-1]], layers,
                                no_uncond_weights, fc_layers=list(range(1, 2*len(layers), 2)))
            hidden_size = layers[-1]
        else:
            hidden_size = in_size

        ### Create fully-connected output-layers ###
        # Note, technically there is no difference between having a separate
        # fully-connected layer per target shape or a single fully-connected
        # layer producing all weights at once (in any case, each output is
        # connceted to all hidden units).
        # I guess it is more computationally efficient to have one output layer
        # and then split the output according to the target shapes.
        self._add_fc_layers([hidden_size], [self.num_outputs],
                            no_uncond_weights, fc_layers=[2*len(layers)+1])

        ### Finalize construction ###
        # All parameters are unconditional except the embeddings created at the
        # very beginning.
        
        self._unconditional_param_shapes_ref = \
            list(range(num_cond_embs, len(self.param_shapes)))

        self._is_properly_setup()

        if verbose:
            print('ICNN Hypernet created.')
            print(self)

    def convexify_tensors(self, tensors):
        """Apply non-negative constraint to each tensor in a nested list."""
        for i in range(len(tensors)):
            for j in range(len(tensors[i])):
                tensors[i][j].clamp_(min=0)
    
    def _flat_to_ret_format(self, flat_out, ret_format):
        """Helper function to convert flat hypernet output into desired output
        format.

        Args:
            flat_out (torch.Tensor): The flat output tensor corresponding to
                ``ret_format='flattened'``.
            ret_format (str): The target output format. See docstring of method
                :meth:`forward`.

        Returns:
            (list or torch.)
        """
        assert ret_format in ['flattened', 'sequential', 'squeezed']
        assert len(flat_out.shape) == 2
        batch_size = flat_out.shape[0]

        if ret_format == 'flattened':
            return flat_out

        ret = [[] for _ in range(batch_size)]
        ind = 0
        for s in self.target_shapes:
            num = int(np.prod(s))

            W = flat_out[:, ind:ind+num]
            W = W.view(batch_size, *s)

            for bind, W_b in enumerate(torch.split(W, 1, dim=0)):
                W_b = torch.squeeze(W_b, dim=0)
                assert np.all(np.equal(W_b.shape, s))
                ret[bind].append(W_b)

            ind += num

        if ret_format == 'squeezed' and batch_size == 1:
            return ret[0]

        return ret

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed',embedding=None):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int, optional): This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        _input_required = True
        if embedding is not None:
            _input_required = False
        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                                          cond_input=cond_input, cond_id=cond_id, weights=weights,
                                          distilled_params=distilled_params, condition=condition,
                                          ret_format=ret_format, _input_required=_input_required)
        # print(uncond_input)
        # print(cond_input)
        # print(uncond_weights)
        ### Prepare hypernet input ###
        assert self._uncond_in_size == 0 or uncond_input is not None
        assert self._cond_in_size == 0 or cond_input is not None or embedding is not None
        if embedding is not None:
            h = embedding
        else:
            if uncond_input is not None:
                assert len(uncond_input.shape) == 2 and \
                    uncond_input.shape[1] == self._uncond_in_size
                h = uncond_input
            if cond_input is not None:
                assert len(cond_input.shape) == 2 and \
                    cond_input.shape[1] == self._cond_in_size
                h = cond_input
            if uncond_input is not None and cond_input is not None:
                h = torch.cat([uncond_input, cond_input], dim=1)

        ### Extract layer weights ###
        bn_scales = []
        bn_shifts = []
        fc_weights = []
        fc_biases = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]
            # print(meta)

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        

        ### Process inputs through network ###
        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)
            
            h = F.linear(h, fc_weights[i], bias=fc_biases[i])

            if not last_layer:
                # Batch-norm
                if self._use_batch_norm:
                    h = self.batchnorm_layers[i].forward(h, running_mean=None,
                                                         running_var=None, weight=bn_scales[i],
                                                         bias=bn_shifts[i], stats_id=condition)

                # Dropout
                if self._dropout_rate != -1:
                    h = self._dropout(h)

                # Non-linearity
                if self._act_fn is not None:
                    h = self._act_fn(h)

            # print(h.shape, fc_weights[i].shape, fc_biases[i].shape)
            
        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)
        # print(ret)

        # ret = self.convexify_tensors(ret)

        return ret

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None
    

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            uncond_var=1., cond_var=1., mnet=None,
                            w_val=None, w_var=None, b_val=None, b_var=None):
        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value "%s" for argument "method".' %
                             method)
        if self.unconditional_params is None:
            assert self._no_uncond_weights
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')

        ### Extract meta-information about target shapes ###
        meta = None
        if mnet is not None:
            assert isinstance(mnet, MainNetInterface)

            try:
                meta = mnet.param_shapes_meta
            except:
                meta = None

            if meta is not None:
                if len(self.target_shapes) == len(mnet.param_shapes):
                    pass
                    # meta = mnet.param_shapes_meta
                elif len(self.target_shapes) == len(mnet.hyper_shapes_learned):
                    meta = []
                    for ii in mnet.hyper_shapes_learned_ref:
                        meta.append(mnet.param_shapes_meta[ii])
                else:
                    warn('Target shapes of this hypernetwork could not be ' +
                         'matched to the meta information provided to the ' +
                         'initialization.')
                    meta = None

        # TODO If the user doesn't (or can't) provide an `mnet` instance, we
        # should alternatively allow him to pass meta information directly.
        if meta is None:
            meta = []

            # Heuristical approach to derive meta information from given shapes.
            layer_ind = 0
            for i, s in enumerate(self.target_shapes):
                curr_meta = dict()

                if len(s) > 1:
                    curr_meta['name'] = 'weight'
                    curr_meta['layer'] = layer_ind
                    layer_ind += 1
                else:  # just a heuristic, we can't know
                    curr_meta['name'] = 'bias'
                    if i > 0 and meta[-1]['name'] == 'weight':
                        curr_meta['layer'] = meta[-1]['layer']
                    else:
                        curr_meta['layer'] = -1

                meta.append(curr_meta)

        assert len(meta) == len(self.target_shapes)

        # Mapping from layer index to the corresponding shape.
        layer_shapes = dict()
        # Mapping from layer index to whether the layer has a bias vector.
        layer_has_bias = defaultdict(lambda: False)
        for i, m in enumerate(meta):
            if m['name'] == 'weight' and m['layer'] != -1:
                assert len(self.target_shapes[i]) > 1
                layer_shapes[m['layer']] = self.target_shapes[i]
            if m['name'] == 'bias' and m['layer'] != -1:
                layer_has_bias[m['layer']] = True

        ### Compute input variance ###
        cond_dim = self._cond_in_size
        uncond_dim = self._uncond_in_size
        inp_dim = cond_dim + uncond_dim

        input_variance = 0
        if cond_dim > 0:
            input_variance += (cond_dim / inp_dim) * cond_var
        if uncond_dim > 0:
            input_variance += (uncond_dim / inp_dim) * uncond_var

        ### Initialize hidden layers to preserve variance ###
        # Note, if batchnorm layers are used, they will simply be initialized to
        # have no effect after initialization. This does not effect the
        # performed whitening operation.
        if self.batchnorm_layers is not None:
            for bn_layer in self.batchnorm_layers:
                if hasattr(bn_layer, 'scale'):
                    nn.init.ones_(bn_layer.scale)
                if hasattr(bn_layer, 'bias'):
                    nn.init.zeros_(bn_layer.bias)

            # Since batchnorm layers whiten the statistics of hidden
            # acitivities, the variance of the input will not be preserved by
            # Xavier/Kaiming.
            if len(self.batchnorm_layers) > 0:
                input_variance = 1.

        # We initialize biases with 0 (see Xavier assumption 4 in the Hyperfan
        # paper). Otherwise, we couldn't ignore the biases when computing the
        # output variance of a layer.
        # Note, we have to use fan-in init for the hidden layer to ensure the
        # property, that we preserve the input variance.
        assert len(self._layers) + 1 == len(self.layer_weight_tensors)
        for i, w_tensor in enumerate(self.layer_weight_tensors[:-1]):
            if use_xavier:
                iutils.xavier_fan_in_(w_tensor)
            else:
                torch.nn.init.kaiming_uniform_(w_tensor, mode='fan_in',
                                               nonlinearity='relu')

            if self.has_bias:
                nn.init.zeros_(self.layer_bias_vectors[i])

        ### Define default parameters of weight init distributions ###
        w_val_list = []
        w_var_list = []
        b_val_list = []
        b_var_list = []

        for i, m in enumerate(meta):
            def extract_val(user_arg):
                curr = None
                if isinstance(user_arg, (list, tuple)) and \
                        user_arg[i] is not None:
                    curr = user_arg[i]
                elif isinstance(user_arg, (dict)) and \
                        m['name'] in user_arg.keys():
                    curr = user_arg[m['name']]
                return curr
            curr_w_val = extract_val(w_val)
            curr_w_var = extract_val(w_var)
            curr_b_val = extract_val(b_val)
            curr_b_var = extract_val(b_var)

            if m['name'] == 'weight' or m['name'] == 'bias':
                if None in [curr_w_val, curr_w_var, curr_b_val, curr_b_var]:
                    # If distribution not fully specified, then we just fall
                    # back to hyper-fan init.
                    curr_w_val = None
                    curr_w_var = None
                    curr_b_val = None
                    curr_b_var = None
            else:
                assert m['name'] in ['bn_scale', 'bn_shift', 'cm_scale',
                                     'cm_shift', 'embedding']
                if curr_w_val is None:
                    curr_w_val = 0
                if curr_w_var is None:
                    curr_w_var = 0
                if curr_b_val is None:
                    curr_b_val = 1 if m['name'] in ['bn_scale', 'cm_scale'] \
                        else 0
                if curr_b_var is None:
                    curr_b_var = 1 if m['name'] in ['embedding'] else 0

            w_val_list.append(curr_w_val)
            w_var_list.append(curr_w_var)
            b_val_list.append(curr_b_val)
            b_var_list.append(curr_b_var)

        ### Initialize output heads ###
        # Note, that all output heads are realized internally via one large
        # fully-connected layer.
        # All output heads are linear layers. The biases of these linear
        # layers (called gamma and beta in the paper) are simply initialized
        # to zero. Note, that we allow deviations from this below.
        if self.has_bias:
            nn.init.zeros_(self.layer_bias_vectors[-1])

        c_relu = 1 if use_xavier else 2

        # We are not interested in the fan-out, since the fan-out is just
        # the number of elements in the main network.
        # `fan-in` is called `d_k` in the paper and is just the size of the
        # last hidden layer.
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.layer_weight_tensors[-1])

        s_ind = 0
        for i, out_shape in enumerate(self.target_shapes):
            m = meta[i]
            e_ind = s_ind + int(np.prod(out_shape))

            curr_w_val = w_val_list[i]
            curr_w_var = w_var_list[i]
            curr_b_val = b_val_list[i]
            curr_b_var = b_var_list[i]

            if curr_w_val is None:
                c_bias = 2 if layer_has_bias[m['layer']] else 1

                if m['name'] == 'bias':
                    m_fan_out = out_shape[0]

                    # NOTE For the hyperfan-out init, we also need to know the
                    # fan-in of the layer.
                    if m['layer'] != -1:
                        m_fan_in, _ = iutils.calc_fan_in_and_out(
                            layer_shapes[m['layer']])
                    else:
                        # FIXME Quick-fix.
                        m_fan_in = m_fan_out

                    var_in = c_relu / (2. * fan_in * input_variance)
                    num = c_relu * (1. - m_fan_in/m_fan_out)
                    denom = fan_in * input_variance
                    var_out = max(0, num / denom)

                else:
                    assert m['name'] == 'weight'
                    m_fan_in, m_fan_out = iutils.calc_fan_in_and_out(out_shape)

                    var_in = c_relu / (c_bias * m_fan_in * fan_in *
                                       input_variance)
                    var_out = c_relu / (m_fan_out * fan_in * input_variance)

                if method == 'in':
                    var = var_in
                elif method == 'out':
                    var = var_out
                elif method == 'harmonic':
                    var = 2 * (1./var_in + 1./var_out)
                else:
                    raise ValueError('Method %s invalid.' % method)

                # Initialize output head weight tensor using `var`.
                std = math.sqrt(var)
                a = math.sqrt(3.0) * std
                torch.nn.init._no_grad_uniform_(
                    self.layer_weight_tensors[-1][s_ind:e_ind, :], -a, a)
            else:
                if curr_w_var == 0:
                    nn.init.constant_(
                        self.layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val)
                else:
                    std = math.sqrt(curr_w_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_(
                        self.layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val-a, curr_w_val+a)

                if curr_b_var == 0:
                    nn.init.constant_(
                        self.layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val)
                else:
                    std = math.sqrt(curr_b_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_(
                        self.layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val-a, curr_b_val+a)

            s_ind = e_ind
    def _preprocess_forward_args(self, _input_required=True,
                                 _parse_cond_id_fct=None, **kwargs):
        """Parse all :meth:`forward` arguments.

        Note:
            This method is currently not considering the arguments
            ``distilled_params`` and ``condition``.

        Args:
            _input_required (bool): Whether at least one of the forward
                arguments ``uncond_input``, ``cond_input`` and ``cond_id`` has
                to be not ``None``.
            _parse_cond_id_fct (func): A function with signature
                ``_parse_cond_id_fct(self, cond_ids, cond_weights)``, where
                ``self`` is the current object, ``cond_ids`` is a ``list`` of
                integers and ``cond_weights`` are the parsed conditional weights
                if any (see return values).
                The function is expected to parse argument ``cond_id`` of the
                :meth:`forward` method. If not provided, we simply use the
                indices within ``cond_id`` to stack elements of
                :attr:`conditional_params`.
            **kwargs: All keyword arguments passed to the :meth:`forward`
                method.

        Returns:
            (tuple): Tuple containing:

            - **uncond_input**: The argument ``uncond_input`` passed to the
              :meth:`forward` method.
            - **cond_input**: If provided, then this is just argument
              ``cond_input`` of the :meth:`forward` method. Otherwise, it is
              either ``None`` or if provided, the conditional input will be
              assembled from the parsed conditional weights ``cond_weights``
              using :meth:`forward` argument ``cond_id``.
            - **uncond_weights**: The unconditional weights :math:`\\theta` to
              be used during forward processing (they will be assembled from
              internal and given weights).
            - **cond_weights**: The conditional weights if tracked be the
              hypernetwork. The parsing is done analoguously as for
              ``uncond_weights``.
        """
        if kwargs['ret_format'] not in ['flattened', 'sequential', 'squeezed']:
            raise ValueError('Return format %s unknown.' \
                             % (kwargs['ret_format']))

        #####################
        ### Parse Weights ###
        #####################
        # We first parse the weights as they night be needed later to choose
        # inputs via `cond_id`.
        uncond_weights = self.unconditional_params
        # print(len(self.internal_params))
        # print('un')
        # print(self._unconditional_param_shapes_ref)
        # print(uncond_weights)
        cond_weights = self.conditional_params
        # print('con')
        # print(cond_weights)
        if kwargs['weights'] is not None:
            if isinstance(kwargs['weights'], dict):
                assert 'uncond_weights' in kwargs['weights'].keys() or \
                       'cond_weights' in kwargs['weights'].keys()

                if 'uncond_weights' in kwargs['weights'].keys():
                    # For simplicity, we assume all unconditional parameters
                    # are passed. This might have to be adapted in the
                    # future.
                    assert len(kwargs['weights']['uncond_weights']) == \
                           len(self.unconditional_param_shapes)
                    uncond_weights = kwargs['weights']['uncond_weights']
                if 'cond_weights' in kwargs['weights'].keys():
                    # Again, for simplicity, we assume all conditional weights
                    # have to be passed.
                    assert len(kwargs['weights']['cond_weights']) == \
                           len(self.conditional_param_shapes)
                    cond_weights = kwargs['weights']['cond_weights']

            else: # list
                if self.hyper_shapes_learned is not None and \
                        len(kwargs['weights']) == \
                        len(self.hyper_shapes_learned):
                    # In this case, we build up conditional and
                    # unconditional weights from internal and given weights.
                    weights = []
                    for i in range(len(self.param_shapes)):
                        if i in self.hyper_shapes_learned_ref:
                            idx = self.hyper_shapes_learned_ref.index(i)
                            weights.append(kwargs['weights'][idx])
                        else:
                            meta = self.param_shapes_meta[i]
                            assert meta['index'] != -1
                            weights.append( \
                                self.internal_params[meta['index']])
                else:
                    if len(kwargs['weights']) != len(self.param_shapes):
                        raise ValueError('The length of argument ' +
                            '"weights" does not meet the specifications.')
                    # In this case, we simply split the given weights into
                    # conditional and unconditional weights.
                    weights = kwargs['weights']
                assert len(weights) == len(self.param_shapes)

                # Split 'weights' into conditional and unconditional weights.
                up_ref = self.unconditional_param_shapes_ref
                cp_ref = self.conditional_param_shapes_ref

                if up_ref is not None:
                    uncond_weights = [None] * len(up_ref)
                else:
                    up_ref = []
                    uncond_weights = None
                if cp_ref is not None:
                    cond_weights = [None] * len(cp_ref)
                else:
                    cp_ref = []
                    cond_weights = None

                for i in range(len(self.param_shapes)):
                    if i in up_ref:
                        idx = up_ref.index(i)
                        assert uncond_weights[idx] is None
                        uncond_weights[idx] = weights[i]
                    else:
                        assert i in cp_ref
                        idx = cp_ref.index(i)
                        assert cond_weights[idx] is None
                        cond_weights[idx] = weights[i]

        ####################
        ### Parse Inputs ###
        ####################
        if _input_required and kwargs['uncond_input'] is None and \
                kwargs['cond_input'] is None and kwargs['cond_id'] is None:
            raise RuntimeError('No hypernet inputs have been provided!')

        # No further preprocessing required.
        uncond_input = kwargs['uncond_input']

        if kwargs['cond_input'] is not None and kwargs['cond_id'] is not None:
            raise ValueError('You cannot provide arguments "cond_input" and ' +
                             '"cond_id" simultaneously!')

        cond_input = None
        if kwargs['cond_input'] is not None:
            cond_input = kwargs['cond_input']
            if len(cond_input.shape) == 1:
                raise ValueError('Batch dimension for conditional inputs is ' +
                                 'missing.')
        if kwargs['cond_id'] is not None:
            assert isinstance(kwargs['cond_id'], (int, list))
            cond_ids = kwargs['cond_id']
            if isinstance(cond_ids, int):
                cond_ids = [cond_ids]

            if _parse_cond_id_fct is not None:
                cond_input = _parse_cond_id_fct(self, cond_ids, cond_weights)
            else:
                if cond_weights is None:
                    raise ValueError('Forward option "cond_id" can only be ' +
                                     'used if conditional parameters are ' 
                                     'maintained internally or passed to the ' +
                                     'forward method via option "weights".')

                assert len(cond_weights) == len(self.conditional_param_shapes)
                if len(cond_weights) != self.num_known_conds:
                    raise RuntimeError('Do not know how to translate IDs to ' +
                                       'conditional inputs.')

                cond_input = []
                for i, cid in enumerate(cond_ids):
                    if cid < 0 or cid >= self.num_known_conds:
                        raise ValueError('Condition %d not existing!' % (cid))

                    cond_input.append(cond_weights[cid])
                    if i > 0:
                        # Assumption when not providing `_parse_cond_id_fct`.
                        assert np.all(np.equal(cond_input[0].shape,
                                               cond_input[i].shape))

                cond_input = torch.stack(cond_input, dim=0)

        # If we are given both, unconditional and conditional inputs, we
        # have to ensure that they use the same batch size.
        if cond_input is not None and uncond_input is not None:
            # We assume the first dimension being the batch dimension.
            # Note, some old hnet implementations could only process one
            # embedding at a time and it was ok to not have a dedicated
            # batch dimension. To avoid nasty bugs we enforce a separate
            # batch dimension.
            assert len(cond_input.shape) > 1 and len(uncond_input.shape) > 1
            if cond_input.shape[0] != uncond_input.shape[0]:
                # If one batch-size is 1, we just repeat the input.
                if cond_input.shape[0] == 1:
                    batch_size = uncond_input.shape[0]
                    cond_input = cond_input.expand(batch_size,
                                                   *cond_input.shape[1:])
                elif uncond_input.shape[0] == 1:
                    batch_size = cond_input.shape[0]
                    uncond_input = uncond_input.expand(batch_size,
                        *uncond_input.shape[1:])
                else:
                    raise RuntimeError('Batch dimensions of hypernet ' +
                                       'inputs do not match!')
            assert cond_input.shape[0] == uncond_input.shape[0]

        return uncond_input, cond_input, uncond_weights, cond_weights

    def get_cond_in_emb(self, cond_id):
        if self.conditional_params is None:
            raise RuntimeError('Input embeddings are not internally ' +
                               'maintained!')
        if not isinstance(cond_id, int) or cond_id < 0 or \
                cond_id >= len(self.conditional_params):
            raise RuntimeError('Option "cond_id" must be between 0 and %d!'
                               % (len(self.conditional_params)-1))
        return self.conditional_params[cond_id]
    
def match_dimensions(tensor_to_match, target_tensor):
    """
    Adjust the dimensions of `tensor_to_match` to match the dimensions of `target_tensor`.
    
    Args:
        tensor_to_match (torch.Tensor): The tensor to adjust.
        target_tensor (torch.Tensor): The tensor with the target dimensions.
        device (torch.device): The device to create new tensors on.

    Returns:
        torch.Tensor: The adjusted tensor with dimensions matching `target_tensor`.
    """
    if tensor_to_match.shape == target_tensor.shape:
        return tensor_to_match

    in_features = tensor_to_match.shape[1]
    out_features = target_tensor.shape[1]
    
    # Creating a weight matrix for dimension matching
    weight = torch.randn(out_features, in_features, device='cuda')
    
    # Adjusting the dimensions
    adjusted_tensor = torch.matmul(tensor_to_match, weight.T)

    return adjusted_tensor



if __name__ == '__main__':
    pass
