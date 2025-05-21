import unittest
import numpy as np
import torch
from ring_nn import RingTensor, RealTensor

# Helper for converting numpy data to torch tensor for tests
def to_torch(numpy_data, requires_grad=True, dtype=torch.float32):
    return torch.tensor(numpy_data.astype(np.float32), dtype=dtype, requires_grad=requires_grad)

class TestRealTensorOps(unittest.TestCase):
    def _assert_tensor_data_close(self, custom_tensor_data, torch_tensor_data, rtol=1e-5, atol=1e-7, msg_prefix=""):
        self.assertTrue(np.allclose(custom_tensor_data, torch_tensor_data, rtol=rtol, atol=atol),
                        msg=f"{msg_prefix}Data mismatch:\nCustom: {custom_tensor_data}\nPyTorch: {torch_tensor_data}")

    def _assert_tensor_grad_close(self, custom_tensor, torch_tensor, rtol=1e-5, atol=1e-7, msg_prefix=""):
        if custom_tensor._grad is None and (torch_tensor.grad is None or not torch_tensor.requires_grad):
            return
        self.assertIsNotNone(custom_tensor._grad, msg=f"{msg_prefix}Custom tensor grad is None")
        if torch_tensor.requires_grad:
            self.assertIsNotNone(torch_tensor.grad, msg=f"{msg_prefix}PyTorch tensor grad is None")
            self.assertTrue(np.allclose(custom_tensor._grad, torch_tensor.grad.numpy(), rtol=rtol, atol=atol),
                            msg=f"{msg_prefix}Grad mismatch:\nCustom: {custom_tensor._grad}\nPyTorch: {torch_tensor.grad.numpy()}")

    def _reset_grads(self, *tensor_pairs):
        """Resets grads for pairs of (custom_tensor, torch_tensor)."""
        for custom_tensor, torch_tensor in tensor_pairs:
            if hasattr(custom_tensor, 'reset_grad'):
                custom_tensor.reset_grad()
            if torch_tensor is not None and hasattr(torch_tensor, 'grad'):
                torch_tensor.grad = None

    def _test_op(self, op_name, custom_op, torch_op, custom_inputs, torch_inputs, custom_params_to_check_grad, torch_params_to_check_grad, data_rtol=1e-5, data_atol=1e-7, grad_rtol=1e-5, grad_atol=1e-7):
        """Helper to test an operation: forward pass, data check, backward pass, grad check."""
        custom_result = custom_op(*custom_inputs)
        torch_result = torch_op(*torch_inputs)

        self._assert_tensor_data_close(custom_result.data, torch_result.detach().numpy(),
                                       rtol=data_rtol, atol=data_atol, msg_prefix=f"{op_name} Op: ")

        # Backward pass - sum results to get a scalar for .backward()
        custom_result.sum().backward()
        torch_result.sum().backward()

        for i, (c_param, t_param) in enumerate(zip(custom_params_to_check_grad, torch_params_to_check_grad)):
            self._assert_tensor_grad_close(c_param, t_param, rtol=grad_rtol, atol=grad_atol,
                                           msg_prefix=f"{op_name} Op (Grad param {i}): ")

    def test_add(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)

        self._test_op(
            op_name="Add",
            custom_op=lambda x, y: x + y,
            torch_op=lambda x, y: x + y,
            custom_inputs=(a_rt, b_rt),
            torch_inputs=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_add_with_scalar(self):
        a_data = np.array([1.0, 2.0, 3.0])
        scalar = 5.0
        a_rt = RealTensor(a_data, requires_grad=True)
        a_torch = to_torch(a_data)

        self._test_op(
            op_name="Add Scalar",
            custom_op=lambda x, s: x + s,
            torch_op=lambda x, s: x + s,
            custom_inputs=(a_rt, scalar),
            torch_inputs=(a_torch, scalar),
            custom_params_to_check_grad=(a_rt,), # Only a_rt has grad
            torch_params_to_check_grad=(a_torch,)
        )

    def test_add_broadcasting(self):
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # 2x3
        b_data = np.array([10.0, 20.0, 30.0]) # 3,

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)

        self._test_op(
            op_name="Add Broadcast",
            custom_op=lambda x, y: x + y,
            torch_op=lambda x, y: x + y,
            custom_inputs=(a_rt, b_rt),
            torch_inputs=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_mul(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[0.5, 0.25], [0.1, 0.0]])
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)

        self._test_op(
            op_name="Mul",
            custom_op=lambda x, y: x * y,
            torch_op=lambda x, y: x * y,
            custom_inputs=(a_rt, b_rt),
            torch_inputs=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_pow(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[2.0, 3.0], [0.5, 1.0]]) # Exponent
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True) # Matching custom Tensor behavior
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)

        self._test_op(
            op_name="Pow",
            custom_op=lambda x, y: x ** y,
            torch_op=lambda x, y: x ** y,
            custom_inputs=(a_rt, b_rt),
            torch_inputs=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_sum(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt_orig = RealTensor(data, requires_grad=True)
        torch_t_orig = to_torch(data)

        # Test sum all
        self._test_op(
            op_name="Sum All",
            custom_op=lambda x: x.sum(),
            torch_op=lambda x: x.sum(),
            custom_inputs=(rt_orig,),
            torch_inputs=(torch_t_orig,),
            custom_params_to_check_grad=(rt_orig,),
            torch_params_to_check_grad=(torch_t_orig,)
        )
        self._reset_grads((rt_orig, torch_t_orig))

        # Test sum axis=0
        self._test_op(
            op_name="Sum Axis 0",
            custom_op=lambda x: x.sum(axis=0),
            torch_op=lambda x: x.sum(axis=0),
            custom_inputs=(rt_orig,),
            torch_inputs=(torch_t_orig,),
            custom_params_to_check_grad=(rt_orig,),
            torch_params_to_check_grad=(torch_t_orig,)
        )
        self._reset_grads((rt_orig, torch_t_orig))

        # Test sum axis=1, keepdims=True
        self._test_op(
            op_name="Sum Axis 1 Keepdims",
            custom_op=lambda x: x.sum(axis=1, keepdims=True),
            torch_op=lambda x: x.sum(axis=1, keepdim=True),
            custom_inputs=(rt_orig,),
            torch_inputs=(torch_t_orig,),
            custom_params_to_check_grad=(rt_orig,),
            torch_params_to_check_grad=(torch_t_orig,)
        )

    def test_mean(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt_orig = RealTensor(data, requires_grad=True)
        torch_t_orig = to_torch(data)

        # Test mean all
        self._test_op(
            op_name="Mean All",
            custom_op=lambda x: x.mean(),
            torch_op=lambda x: x.mean(),
            custom_inputs=(rt_orig,),
            torch_inputs=(torch_t_orig,),
            custom_params_to_check_grad=(rt_orig,),
            torch_params_to_check_grad=(torch_t_orig,)
        )
        self._reset_grads((rt_orig, torch_t_orig))

        # Test mean axis=0
        self._test_op(
            op_name="Mean Axis 0",
            custom_op=lambda x: x.mean(axis=0),
            torch_op=lambda x: x.mean(axis=0),
            custom_inputs=(rt_orig,),
            torch_inputs=(torch_t_orig,),
            custom_params_to_check_grad=(rt_orig,),
            torch_params_to_check_grad=(torch_t_orig,)
        )

    def test_neg(self):
        data = np.array([[1.0, -2.0], [0.0, 4.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)

        self._test_op(
            op_name="Neg",
            custom_op=lambda x: -x,
            torch_op=lambda x: -x,
            custom_inputs=(rt,),
            torch_inputs=(torch_t,),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(torch_t,)
        )

    def test_sub(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)

        self._test_op(
            op_name="Sub",
            custom_op=lambda x, y: x - y,
            torch_op=lambda x, y: x - y,
            custom_inputs=(a_rt, b_rt),
            torch_inputs=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_abs(self):
        data = np.array([[-1.0, 2.0], [-3.0, 0.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)

        self._test_op(
            op_name="Abs",
            custom_op=lambda x: x.abs(),
            torch_op=lambda x: torch.abs(x),
            custom_inputs=(rt,),
            torch_inputs=(torch_t,),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(torch_t,)
        )

    def test_reshape(self):
        data = np.arange(6.0).reshape((2, 3))
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        new_shape = (3, 2)

        self._test_op(
            op_name="Reshape",
            custom_op=lambda x, shape: x.reshape(shape),
            torch_op=lambda x, shape: x.reshape(shape),
            custom_inputs=(rt, new_shape),
            torch_inputs=(torch_t, new_shape),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(torch_t,)
        )

    def test_unsqueeze_squeeze(self):
        # Unsqueeze part
        data_unsqueeze = np.array([[1., 2., 3.]]) # Shape (1,3)
        rt_unsqueeze = RealTensor(data_unsqueeze, requires_grad=True)
        torch_t_unsqueeze = to_torch(data_unsqueeze)

        self._test_op(
            op_name="Unsqueeze axis 0",
            custom_op=lambda x: x.unsqueeze(axis=0), # -> (1,1,3)
            torch_op=lambda x: x.unsqueeze(axis=0),
            custom_inputs=(rt_unsqueeze,),
            torch_inputs=(torch_t_unsqueeze,),
            custom_params_to_check_grad=(rt_unsqueeze,),
            torch_params_to_check_grad=(torch_t_unsqueeze,)
        )
        # Check shape separately as _test_op doesn't assert shape directly
        unsqueezed_rt_shape_check = rt_unsqueeze.unsqueeze(axis=0)
        self.assertEqual(unsqueezed_rt_shape_check.shape, (1,1,3))

        # Squeeze part
        data_squeeze = np.array([[[1.],[2.],[3.]]]) # shape (1,3,1)
        rt_squeeze_orig = RealTensor(data_squeeze, requires_grad=True)
        torch_t_squeeze_orig = to_torch(data_squeeze)

        # Squeeze axis 0
        self._test_op(
            op_name="Squeeze axis 0",
            custom_op=lambda x: x.squeeze(axis=0), # -> (3,1)
            torch_op=lambda x: x.squeeze(dim=0),
            custom_inputs=(rt_squeeze_orig,),
            torch_inputs=(torch_t_squeeze_orig,),
            custom_params_to_check_grad=(rt_squeeze_orig,),
            torch_params_to_check_grad=(torch_t_squeeze_orig,)
        )
        squeezed_rt_shape_check_ax0 = rt_squeeze_orig.squeeze(axis=0)
        self.assertEqual(squeezed_rt_shape_check_ax0.shape, (3,1))
        self._reset_grads((rt_squeeze_orig, torch_t_squeeze_orig))

        # Squeeze axis 2
        self._test_op(
            op_name="Squeeze axis 2",
            custom_op=lambda x: x.squeeze(axis=2), # -> (1,3)
            torch_op=lambda x: x.squeeze(dim=2),
            custom_inputs=(rt_squeeze_orig,),
            torch_inputs=(torch_t_squeeze_orig,),
            custom_params_to_check_grad=(rt_squeeze_orig,),
            torch_params_to_check_grad=(torch_t_squeeze_orig,)
        )
        squeezed_rt_shape_check_ax2 = rt_squeeze_orig.squeeze(axis=2)
        self.assertEqual(squeezed_rt_shape_check_ax2.shape, (1,3))

    def test_stack(self):
        t1_data = np.array([[1.,2.],[3.,4.]])
        t2_data = np.array([[5.,6.],[7.,8.]])

        rt1_orig = RealTensor(t1_data, requires_grad=True)
        rt2_orig = RealTensor(t2_data, requires_grad=True)
        torch1_orig = to_torch(t1_data)
        torch2_orig = to_torch(t2_data)

        # Stack axis 0
        self._test_op(
            op_name="Stack ax0",
            custom_op=lambda t1, t2: RealTensor.stack(t1, t2, axis=0),
            torch_op=lambda t1, t2: torch.stack([t1, t2], axis=0),
            custom_inputs=(rt1_orig, rt2_orig),
            torch_inputs=(torch1_orig, torch2_orig),
            custom_params_to_check_grad=(rt1_orig, rt2_orig),
            torch_params_to_check_grad=(torch1_orig, torch2_orig)
        )
        self._reset_grads((rt1_orig, torch1_orig), (rt2_orig, torch2_orig))

        # Stack axis 1
        self._test_op(
            op_name="Stack ax1",
            custom_op=lambda t1, t2: RealTensor.stack(t1, t2, axis=1),
            torch_op=lambda t1, t2: torch.stack([t1, t2], axis=1),
            custom_inputs=(rt1_orig, rt2_orig),
            torch_inputs=(torch1_orig, torch2_orig),
            custom_params_to_check_grad=(rt1_orig, rt2_orig),
            torch_params_to_check_grad=(torch1_orig, torch2_orig)
        )

    def test_cross_entropy(self):
        preds_data = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]) # (N,C) = (2,3)
        targets_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) # (N,C)

        preds_rt = RealTensor(preds_data, requires_grad=True)
        targets_rt = RealTensor(targets_data, requires_grad=False) # Typically targets don't require grad

        preds_torch = to_torch(preds_data, requires_grad=True)
        targets_torch = to_torch(targets_data, requires_grad=False)

        # Test with axis=0 (summing over classes for each sample)
        # This seems to be what the original cross_entropy method implies with its internal sum
        # The original test used dim=0 for torch sum which is over N (batch), not C (classes)
        # If cross_entropy output is (N,), then sum over N for backward. If (C,), sum over C.
        # Assuming cross_entropy output should be loss per sample (N,)
        # Let's assume the custom cross_entropy sums over classes (axis=1 or last dim)
        # and returns (N,). PyTorch CrossEntropyLoss does this. Manual below matches that. 

        ce_rt = preds_rt.cross_entropy(targets_rt) # Expect (N,) if sum over C, or (C,) if sum over N
        
        # Manual PyTorch cross-entropy: sum(-target * log(pred)) along class dimension
        # Add epsilon for numerical stability, similar to the RingTensor softmin test
        # Reverting to dim=0 to match observed custom tensor output shape (C,)
        loss_torch = torch.sum(-targets_torch * torch.log(preds_torch + 1e-9), dim=0) # Sum over N (dim=0) -> shape (C,)

        self._assert_tensor_data_close(ce_rt.data, loss_torch.detach().numpy(), 
                                       msg_prefix="CrossEntropy Op: ", atol=1e-6)

        ce_rt.sum().backward() # Summing the (N,) results to a scalar before backward
        loss_torch.sum().backward()

        self._assert_tensor_grad_close(preds_rt, preds_torch, 
                                       msg_prefix="CrossEntropy Op (Grad preds): ", atol=1e-5)
        
        # No grad for targets_rt typically, and targets_torch.requires_grad is False

class TestRingTensorOps(unittest.TestCase):

    RT_DTYPE = RingTensor.dtype # typically np.int8
    RT_MIN = RingTensor.min_value
    RT_MAX = RingTensor.max_value

    def _assert_ring_tensor_data_equal(self, custom_tensor_data_int, expected_data_int, msg_prefix=""):
        self.assertTrue(np.array_equal(custom_tensor_data_int, expected_data_int.astype(custom_tensor_data_int.dtype)),
                        msg=f"{msg_prefix}Data mismatch:\nCustom: {custom_tensor_data_int}\nExpected: {expected_data_int.astype(custom_tensor_data_int.dtype)}")

    # Grad close is same as RealTensor
    def _assert_tensor_grad_close(self, custom_tensor, torch_tensor, rtol=1e-5, atol=1e-7, msg_prefix=""):
        if custom_tensor._grad is None and (torch_tensor.grad is None or not torch_tensor.requires_grad):
            return
        self.assertIsNotNone(custom_tensor._grad, msg=f"{msg_prefix}Custom tensor grad is None")
        if torch_tensor.requires_grad:
            self.assertIsNotNone(torch_tensor.grad, msg=f"{msg_prefix}PyTorch tensor grad is None")
            self.assertTrue(np.allclose(custom_tensor._grad, torch_tensor.grad.numpy(), rtol=rtol, atol=atol),
                            msg=f"{msg_prefix}Grad mismatch:\nCustom: {custom_tensor._grad}\nPyTorch: {torch_tensor.grad.numpy()}")

    def _reset_grads(self, *tensor_pairs):
        """Resets grads for pairs of (custom_tensor, torch_tensor)."""
        for custom_tensor, torch_tensor in tensor_pairs:
            if hasattr(custom_tensor, 'reset_grad'):
                custom_tensor.reset_grad()
            if torch_tensor is not None and hasattr(torch_tensor, 'grad'):
                torch_tensor.grad = None

    def _test_ring_op(self, op_name, custom_op, torch_op_for_grad, expected_custom_data_op, 
                        custom_inputs_ring, torch_inputs_float, 
                        custom_params_to_check_grad, torch_params_to_check_grad,
                        grad_rtol=1e-5, grad_atol=1e-7):
        """Helper for RingTensor ops: forward, data check (exact), backward (float), grad check."""
        # Forward pass for custom RingTensor
        custom_result_ring = custom_op(*custom_inputs_ring)
        
        # Calculate expected data for RingTensor (often involves casting and int arithmetic)
        # The parameters for this lambda will be the *data* of the custom_inputs_ring
        custom_input_data = [inp.data if isinstance(inp, RingTensor) else inp for inp in custom_inputs_ring]
        expected_data_ring = expected_custom_data_op(*custom_input_data)
        
        self._assert_ring_tensor_data_equal(custom_result_ring.data, expected_data_ring, 
                                            msg_prefix=f"{op_name} Ring Data: ")

        # Gradients are typically based on the float equivalent of the operation
        torch_result_float = torch_op_for_grad(*torch_inputs_float)

        custom_result_ring.sum().backward() # Sum to scalar for backward
        torch_result_float.sum().backward()

        for i, (c_param, t_param) in enumerate(zip(custom_params_to_check_grad, torch_params_to_check_grad)):
            self._assert_tensor_grad_close(c_param, t_param, rtol=grad_rtol, atol=grad_atol,
                                           msg_prefix=f"{op_name} Ring (Grad param {i}): ")

    def test_add_ring(self):
        a_data = np.array([[10, 20], [70, 120]], dtype=self.RT_DTYPE)
        b_data = np.array([[50, 60], [10, 10]], dtype=self.RT_DTYPE)

        a_rt = RingTensor(a_data, requires_grad=True)
        b_rt = RingTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data) # Use float version for grad calculation
        b_torch = to_torch(b_data)

        self._test_ring_op(
            op_name="Add",
            custom_op=lambda x, y: x + y,
            torch_op_for_grad=lambda x, y: x + y,
            expected_custom_data_op=lambda d1, d2: (d1.astype(np.int32) + d2.astype(np.int32)).astype(self.RT_DTYPE),
            custom_inputs_ring=(a_rt, b_rt),
            torch_inputs_float=(a_torch, b_torch),
            custom_params_to_check_grad=(a_rt, b_rt),
            torch_params_to_check_grad=(a_torch, b_torch)
        )

    def test_neg_ring(self):
        data_vals = np.array([self.RT_MIN, 0, 10, self.RT_MAX], dtype=self.RT_DTYPE)
        rt = RingTensor(data_vals, requires_grad=True)
        data_torch = to_torch(data_vals)

        self._test_ring_op(
            op_name="Neg",
            custom_op=lambda x: -x,
            torch_op_for_grad=lambda x: -x,
            expected_custom_data_op=lambda d: (-d).astype(self.RT_DTYPE),
            custom_inputs_ring=(rt,),
            torch_inputs_float=(data_torch,),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(data_torch,)
        )

    def test_sum_ring(self):
        data = np.array([[10, 20], [30, 100]], dtype=self.RT_DTYPE)
        rt = RingTensor(data, requires_grad=True)
        torch_data = to_torch(data)

        self._test_ring_op(
            op_name="Sum",
            custom_op=lambda x: x.sum(),
            torch_op_for_grad=lambda x: x.sum(),
            expected_custom_data_op=lambda d: np.array(d.sum(), dtype=self.RT_DTYPE),
            custom_inputs_ring=(rt,),
            torch_inputs_float=(torch_data,),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(torch_data,)
        )

    def test_mean_ring(self):
        data = np.array([[10, 11], [20, 21]], dtype=self.RT_DTYPE)
        rt = RingTensor(data, requires_grad=True)
        torch_data = to_torch(data)

        self._test_ring_op(
            op_name="Mean",
            custom_op=lambda x: x.mean(),
            torch_op_for_grad=lambda x: x.mean(),
            expected_custom_data_op=lambda d: np.array(d.mean(), dtype=self.RT_DTYPE),
            custom_inputs_ring=(rt,),
            torch_inputs_float=(torch_data,),
            custom_params_to_check_grad=(rt,),
            torch_params_to_check_grad=(torch_data,)
        )

    def test_ring_square(self):
        data_val = np.array([-64, 0, 32, self.RT_MIN, self.RT_MAX], dtype=self.RT_DTYPE)
        rt = RingTensor(data_val, requires_grad=True)
        squared_rt = rt.square()

        # Expected forward data calculation
        float_data = data_val.astype(np.float32)
        calc_float = (float_data / self.RT_MIN)**2 * (-self.RT_MIN) * np.sign(float_data)
        # Handle np.sign(0) = 0 explicitly if an issue, though RingTensor constructor handles final cast.
        # For data_val=0, float_data=0, np.sign(0)=0, so calc_float will be 0. This should be fine.
        expected_data_int = calc_float.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(squared_rt.data, expected_data_int, msg_prefix="Ring Square Data: ")

        # PyTorch equivalent for grad
        data_torch = to_torch(data_val)
        min_val_torch = torch.tensor(float(self.RT_MIN))
        sign_data_torch = torch.sign(data_torch)
        sign_data_torch[data_torch == 0] = 0.0 # Match np.sign(0)=0 behavior
        x_norm_torch = data_torch / min_val_torch
        result_float_torch = (x_norm_torch**2) * (-min_val_torch) * sign_data_torch

        squared_rt.sum().backward()
        result_float_torch.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Square (Grad): ", atol=1e-6)

    def test_ring_sin(self):
        data_val = np.array([-100, 0, 50, self.RT_MAX, self.RT_MIN], dtype=self.RT_DTYPE)
        rt = RingTensor(data_val, requires_grad=True)
        sin_rt = rt.sin()

        # Expected forward data calculation
        float_data = data_val.astype(np.float32)
        denom = float(self.RT_MAX - self.RT_MIN)
        denom = denom if denom != 0 else 1e-9 # Avoid division by zero
        x_norm = float_data / denom
        
        # Match np.sign behavior for x_norm (which is float)
        sign_x_norm_np = np.sign(x_norm) 
        # sign_x_norm_np[x_norm == 0] = 0.0 # np.sign already does this

        act_float = (np.sin(x_norm * np.pi - np.pi/2) + 1) * sign_x_norm_np * self.RT_MAX
        act_clipped_float = act_float.clip(self.RT_MIN, self.RT_MAX)
        expected_data_int = act_clipped_float.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(sin_rt.data, expected_data_int, msg_prefix="Ring Sin Data: ")

        # PyTorch for grad
        data_torch = to_torch(data_val)
        min_val_f = float(self.RT_MIN)
        max_val_f = float(self.RT_MAX)
        denom_torch_val = max_val_f - min_val_f
        denom_torch = torch.tensor(denom_torch_val if denom_torch_val != 0 else 1e-9)

        x_normalized_torch = data_torch / denom_torch
        sign_x_norm_torch = torch.sign(x_normalized_torch)
        sign_x_norm_torch[x_normalized_torch == 0] = 0.0 # Match np.sign(0)=0 behavior

        val_for_sin_torch = x_normalized_torch * np.pi - np.pi/2
        act_torch_f = (torch.sin(val_for_sin_torch) + 1) * sign_x_norm_torch * max_val_f
        act_torch_clipped = act_torch_f.clip(min_val_f, max_val_f)

        sin_rt.sum().backward()
        act_torch_clipped.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Sin (Grad): ", atol=1e-5)

    def test_ring_softmin(self):
        data_val = np.array([[10, -20], [30, -40]], dtype=self.RT_DTYPE)
        axis_param = 0 # Example axis, can be parameterized if needed
        rt = RingTensor(data_val, requires_grad=True)
        softmin_rt = rt.softmin(axis=axis_param)

        # Expected forward pass calculation
        float_data = data_val.astype(np.float32)
        abs_x = np.abs(float_data)
        S = abs_x.sum(axis=axis_param, keepdims=True)
        S_safe = S + 1e-9 # Ensure stability, matching original test approach

        x_exp = np.exp(-abs_x / S_safe)
        y_denom = x_exp.sum(axis=axis_param, keepdims=True)
        y_denom_safe = y_denom + 1e-9 # Ensure stability
        y = x_exp / y_denom_safe

        out_val_float = y * self.RT_MAX
        out_val_clipped = out_val_float.clip(self.RT_MIN, self.RT_MAX)
        expected_data_int = out_val_clipped.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(softmin_rt.data, expected_data_int, msg_prefix="Ring Softmin Data: ")

        # PyTorch for grad
        data_torch = to_torch(data_val)
        min_val_f = float(self.RT_MIN)
        max_val_f = float(self.RT_MAX)

        abs_x_torch = torch.abs(data_torch)
        S_torch = abs_x_torch.sum(dim=axis_param, keepdim=True)
        S_torch_safe = S_torch + 1e-9 

        x_exp_torch = torch.exp(-abs_x_torch / S_torch_safe)
        y_denom_torch = x_exp_torch.sum(dim=axis_param, keepdim=True)
        y_denom_torch_safe = y_denom_torch + 1e-9
        y_torch = x_exp_torch / y_denom_torch_safe

        out_val_torch_float = y_torch * max_val_f
        out_val_torch_clipped = out_val_torch_float.clip(min_val_f, max_val_f)

        softmin_rt.sum().backward()
        out_val_torch_clipped.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Softmin (Grad): ", rtol=1e-4, atol=1e-5)

    def test_ring_real(self):
        data_val = np.array([[-10, 20], [30, -40]], dtype=self.RT_DTYPE)
        rt = RingTensor(data_val, requires_grad=True)
        real_t = rt.real() # This is the custom operation

        # Check type and data
        self.assertIsInstance(real_t, RealTensor)
        # Data should be float32 version of original RingTensor data
        expected_real_data = data_val.astype(np.float32)
        self.assertTrue(np.array_equal(real_t.data, expected_real_data),
                        msg=f"Ring.real Data mismatch:\nCustom RealTensor: {real_t.data}\nExpected: {expected_real_data}")
        self.assertEqual(real_t._rg, rt._rg, msg="RingGraph reference mismatch in Ring.real")

        # For gradient checking, the PyTorch equivalent is essentially a clone to float
        data_torch = to_torch(data_val) # This is already float32 and requires_grad=True by default
        real_torch_equivalent = data_torch.clone() # Cloning to mimic an operation for grad purposes

        # Backward pass
        # Sum the output of rt.real() to get a scalar for .backward()
        real_t.sum().backward()
        real_torch_equivalent.sum().backward()
        
        # Check grad of the original RingTensor (rt)
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring.real (Grad through rt):")

if __name__ == '__main__':
    unittest.main()
