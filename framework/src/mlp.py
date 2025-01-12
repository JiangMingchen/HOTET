from hypnettorch.mnets.mlp import MLP
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMLP(MLP):
    
    def __init__(self, n_in=1, n_out=1, hidden_layers=(10, 10),
                activation_fn=torch.nn.ReLU(), use_bias=True, no_weights=False,
                init_weights=None, dropout_rate=-1, use_spectral_norm=False,
                use_batch_norm=False, bn_track_stats=True,
                distill_bn_stats=False, use_context_mod=False,
                context_mod_inputs=False, no_last_layer_context_mod=False,
                context_mod_no_weights=False,
                context_mod_post_activation=False,
                context_mod_gain_offset=False, context_mod_gain_softplus=False,
                out_fn=None, verbose=True):
        
        MLP.__init__(self, n_in, n_out, hidden_layers,
                 activation_fn, use_bias, no_weights,
                 init_weights, dropout_rate, use_spectral_norm,
                 use_batch_norm, bn_track_stats,
                 distill_bn_stats, use_context_mod,
                 context_mod_inputs, no_last_layer_context_mod,
                 context_mod_no_weights,
                 context_mod_post_activation,
                 context_mod_gain_offset, context_mod_gain_softplus,
                 out_fn, verbose)
    
    def push(self, input, weights=None):
        output = autograd.grad(
            outputs=self.forward(input, weights=weights), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output
    
    # for plotting, add new function push_nograd
    # it is referenced from orginal DenseICNN in icnn.py.
    # see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/icnn.py line 158.
    def push_nograd(self, input, weights = None):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        output.data = self.push(input, weights).data
        return output

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): If a list of parameter tensors is given and
                context modulation is used (see argument ``use_context_mod`` in
                constructor), then these parameters are interpreted as context-
                modulation parameters if the length of ``weights`` equals
                :code:`2*len(net.context_mod_layers)`. Otherwise, the length is
                expected to be equal to the length of the attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                Alternatively, a dictionary can be passed with the possible
                keywords ``internal_weights`` and ``mod_weights``. Each keyword
                is expected to map onto a list of tensors.
                The keyword ``internal_weights`` refers to all weights of this
                network except for the weights of the context-modulation layers.
                The keyword ``mod_weights``, on the other hand, refers
                specifically to the weights of the context-modulation layers.
                It is not necessary to specify both keywords.
            distilled_params: Will be passed as ``running_mean`` and
                ``running_var`` arguments of method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.
            condition (int or dict, optional): If ``int`` is provided, then this
                argument will be passed as argument ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.

                If a ``dict`` is provided instead, the following keywords are
                allowed:

                    - ``bn_stats_id``: Will be handled as ``stats_id`` of the
                      batchnorm layers as described above.
                    - ``cmod_ckpt_id``: Will be passed as argument ``ckpt_id``
                      to the method
                      :meth:`utils.context_mod_layer.ContextModLayer.forward`.

        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network.
            - **h_y** (optional): If ``out_fn`` was specified in the
              constructor, then this value will be returned. It is the last
              hidden activation (before the ``out_fn`` has been applied).
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        n_cm = self._num_context_mod_shapes()

        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert('internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys())
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert(len(weights) == len(self.param_shapes))
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert(len(cm_weights) == len(self._context_mod_shapes))
            int_shapes = self.param_shapes[n_cm:]
            assert(len(int_weights) == len(int_shapes))
            for i, s in enumerate(int_shapes):
                assert(np.all(np.equal(s, list(int_weights[i].shape))))

        cm_ind = 0
        bn_ind = 0

        if self._use_batch_norm:
            n_bn = 2 * len(self.batchnorm_layers)
            bn_weights = int_weights[:n_bn]
            layer_weights = int_weights[n_bn:]
        else:
            layer_weights = int_weights

        w_weights = []
        b_weights = []
        for i, p in enumerate(layer_weights):
            if self.has_bias and i % 2 == 1:
                b_weights.append(p)
            else:
                w_weights.append(p)

        ########################
        ### Parse condition ###
        #######################

        bn_cond = None
        cmod_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                assert('bn_stats_id' in condition.keys() or \
                       'cmod_ckpt_id' in condition.keys())
                if 'bn_stats_id' in condition.keys():
                    bn_cond = condition['bn_stats_id']
                if 'cmod_ckpt_id' in condition.keys():
                    cmod_cond = condition['cmod_ckpt_id']

                    # FIXME We always require context-mod weight above, but
                    # we can't pass both (a condition and weights) to the
                    # context-mod layers.
                    # An unelegant solution would be, to just set all
                    # context-mod weights to None.
                    raise NotImplementedError('CM-conditions not implemented!')
            else:
                bn_cond = condition

        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            nn = len(self._batchnorm_layers)
            running_means = [None] * nn
            running_vars = [None] * nn

        if distilled_params is not None:
            if not self._distill_bn_stats:
                raise ValueError('Argument "distilled_params" can only be ' +
                                 'provided if the return value of ' +
                                 'method "distillation_targets()" is not None.')
            shapes = self.hyper_shapes_distilled
            assert(len(distilled_params) == len(shapes))
            for i, s in enumerate(shapes):
                assert(np.all(np.equal(s, list(distilled_params[i].shape))))

            # Extract batchnorm stats from distilled_params
            for i in range(0, len(distilled_params), 2):
                running_means[i//2] = distilled_params[i]
                running_vars[i//2] = distilled_params[i+1]

        elif self._use_batch_norm and self._bn_track_stats and \
                bn_cond is None:
            for i, bn_layer in enumerate(self._batchnorm_layers):
                running_means[i], running_vars[i] = bn_layer.get_stats()

        ###########################
        ### Forward Computation ###
        ###########################
        hidden = x

        # Context-dependent modulation of inputs directly.
        if self._use_context_mod and self._context_mod_inputs:
            hidden = self._context_mod_layers[cm_ind].forward(hidden,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)
            cm_ind += 1

        for l in range(len(w_weights)):
            W = w_weights[l]
            if self.has_bias:
                b = b_weights[l]
            else:
                b = None

            # Linear layer.
            hidden = self._spec_norm(F.linear(hidden, W, bias=b))
            x_res = hidden
            # Only for hidden layers.
            if l < len(w_weights) - 1:
                # Context-dependent modulation (pre-activation).
                if self._use_context_mod and \
                        not self._context_mod_post_activation:
                    hidden = self._context_mod_layers[cm_ind].forward(hidden,
                        weights=cm_weights[2*cm_ind:2*cm_ind+2],
                        ckpt_id=cmod_cond)
                    cm_ind += 1

                # Batch norm
                if self._use_batch_norm:
                    hidden = self._batchnorm_layers[bn_ind].forward(hidden,
                        running_mean=running_means[bn_ind],
                        running_var=running_vars[bn_ind],
                        weight=bn_weights[2*bn_ind],
                        bias=bn_weights[2*bn_ind+1], stats_id=bn_cond)
                    bn_ind += 1

                # Dropout
                if self._dropout_rate != -1:
                    hidden = self._dropout(hidden)

                # Non-linearity
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

                # Context-dependent modulation (post-activation).
                if self._use_context_mod and self._context_mod_post_activation:
                    hidden = self._context_mod_layers[cm_ind].forward(hidden,
                        weights=cm_weights[2*cm_ind:2*cm_ind+2],
                        ckpt_id=cmod_cond)
                    cm_ind += 1

        # Context-dependent modulation in output layer.
        if self._use_context_mod and not self._no_last_layer_context_mod:
            hidden = self._context_mod_layers[cm_ind].forward(hidden,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)
            
        hidden = hidden + x_res
        
        if self._out_fn is not None:
            return self._out_fn(hidden), hidden

        return hidden
