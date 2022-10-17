# ------------------------------------------------------------------------
# Some classes or methods are made by modifying torch.autograd.functional (https://github.com/pytorch/pytorch/blob/master/torch/autograd/functional.py).
# License and Copyright of the partial of the following codes are available at (https://github.com/pytorch/pytorch/blob/master/LICENSE), which are directly copied as follows.
# ======================================================================
# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ======================================================================
# ------------------------------------------------------------------------
import numpy as np
import torch
from torch.autograd.functional import _as_tuple, _check_requires_grad, _grad_postprocess, _autograd_grad, _construct_standard_basis_for, Tuple, _vmap, List, _grad_preprocess, _validate_v, _fill_in_zeros, _tuple_postprocess


def deco_tuple_process(func):
    def wrapper(outputs, inputs, *args, **kwargs):
        inputs, numels = _normalize_potentially_nexted_tuples(inputs)
        outputs = func(outputs, inputs, *args, **kwargs)
        return _recover_potentially_nested_tuples(outputs, numels)

    return wrapper


mygrad = deco_tuple_process(torch.autograd.grad)


@deco_tuple_process
def myjvp(outputs, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the
    Jacobian of the given function at the point given by the inputs.

    Modified from torch.autograd.functional.vjp as we use transposed definition of jacobian

    Args:
        outputs (tuple of Tensors or Tensor):
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")

        # outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the "
                                   "user-provided function returns "
                                   "a single Tensor with a single element.")

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(vjp, is_inputs_tuple)


def myjacobian(outputs, inputs, retain_graph=True, create_graph=False, strict=False, vectorize=False):
    r"""Function that computes the Jacobian of a given function.

    Args:
        outputs (tuple of Tensors or Tensor): a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental, please use at
            your own risk. When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we use the vmap prototype feature as the backend to
            vectorize calls to ``autograd.grad`` so we only invoke it once
            instead of once per row. This should lead to performance
            improvements in many use cases, however, due to this feature
            being incomplete, there may be performance cliffs. Please
            use `torch._C._debug_only_display_vmap_fallback_warnings(True)`
            to show any performance warnings and file us issues if
            warnings exist for your use case. Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input.
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        # inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs,
                                              "outputs of the user-provided function",
                                              "jacobian")

        _check_requires_grad(outputs, "outputs", strict=strict)

        if vectorize:
            if strict:
                raise RuntimeError('torch.autograd.functional.jacobian: `strict=True` '
                                   'and `vectorized=True` are not supported together. '
                                   'Please either set `strict=False` or '
                                   '`vectorize=False`.')
            # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
            #
            # Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
            # It turns out we can compute the jacobian of this function with a single
            # call to autograd.grad by using vmap over the correct grad_outputs.
            #
            # Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
            # into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
            #
            # To get the first row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
            # To get the 2nd row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
            # and so on.
            #
            # Using vmap, we can vectorize all 4 of these computations into one by
            # passing the standard basis for R^4 as the grad_output.
            # vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
            #
            # Now, how do we compute the jacobian *without stacking the output*?
            # We can just split the standard basis across the outputs. So to
            # compute the jacobian of f(x), we'd use
            # >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
            # The grad_outputs looks like the following:
            # ( torch.tensor([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]),
            #   torch.tensor([[0],
            #                 [0],
            #                 [0],
            #                 [1]]) )
            #
            # But we're not done yet!
            # >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
            # returns a Tensor of shape [4, 3]. We have to remember to split the
            # jacobian of shape [4, 3] into two:
            # - one of shape [3, 3] for the first output
            # - one of shape [   3] for the second output

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(_autograd_grad(flat_outputs, inputs, grad_output, retain_graph=retain_graph, create_graph=create_graph))
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx])
                return tuple(vj)

            jacobians_of_flat_output = _vmap(vjp)(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(jac.split(output_numels, dim=0), outputs):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    # transpose
                    dims_permutate = np.roll(np.arange(output_j.ndim + input_i.ndim), output_j.ndim)
                    jacobian_input_i_output_j = jacobian_input_i_output_j.permute(*dims_permutate)

                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            jacobian_input_output = _grad_postprocess(jacobian_input_output, create_graph)
            return _tuple_postprocess(jacobian_input_output, (is_inputs_tuple, is_outputs_tuple))

        jacobian: Tuple[torch.Tensor, ...] = tuple()

        for i, out in enumerate(outputs):

            # mypy complains that expression and variable have different types due to the empty list
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in range(out.nelement()):
                vj = _autograd_grad((out.reshape(-1)[j],), inputs,
                                    retain_graph=True, create_graph=create_graph)

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = ("The jacobian of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode when create_graph=True.".format(i))
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = ("Output {} of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode.".format(i, el_idx))
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            # transposed version
            jacobian += (tuple(torch.stack(jac_i_el, dim=-1).view(inputs[el_idx].size()
                                                                  + out.size()) for (el_idx, jac_i_el) in enumerate(jac_i)),)

        # The outer List corresponds to the number of outputs,
        # the inner List corresponds to the number of inputs.
        # We need to exchange the order of these and convert to tuples
        # before returning.
        jacobian = tuple(zip(*jacobian))

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_inputs_tuple, is_outputs_tuple))


def _normalize_potentially_nexted_tuples(inputs):
    if not isinstance(inputs, tuple):
        return inputs, 1
    else:
        outputs = []
        numels = []
        for el in inputs:
            if isinstance(el, tuple):
                outputs.extend(el)
                numels.append(len(el))
            else:
                outputs.append(el)
                numels.append(0)
        return tuple(outputs), tuple(numels)


def _recover_potentially_nested_tuples(inputs, numels):
    if not isinstance(numels, tuple):
        return inputs
    else:
        idx = 0
        outputs = []
        for nel in numels:
            if nel == 0:
                outputs.append(inputs[idx])
                idx += 1
            else:
                outputs.append(inputs[idx:idx + nel])
                idx += nel

        assert idx == len(inputs), (idx, len(inputs))
        return tuple(outputs)


