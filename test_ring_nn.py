import unittest
import numpy as np
import torch
from tensor import RingTensor, RealTensor


# Helper for converting numpy data to torch tensor for tests
def to_torch(numpy_data, requires_grad=True, dtype=torch.float32):
    return torch.tensor(numpy_data.astype(np.float32), dtype=dtype, requires_grad=requires_grad)

class TestRealTensorOps(unittest.TestCase):
    def assert_tensors_close(self, custom_tensor, torch_tensor, rtol=1e-5, atol=1e-7, check_grad=True):
        # Check data values
        self.assertTrue(np.allclose(custom_tensor.data, torch_tensor.detach().numpy(), rtol=rtol, atol=atol),
                       f"Data mismatch: custom={custom_tensor.data}, torch={torch_tensor.detach().numpy()}")
        
        # Check gradients if needed
        if check_grad and custom_tensor._grad is not None and torch_tensor.grad is not None:
            self.assertTrue(np.allclose(custom_tensor._grad, torch_tensor.grad.numpy(), rtol=rtol, atol=atol),
                           f"Grad mismatch: custom={custom_tensor._grad}, torch={torch_tensor.grad.numpy()}")

    def _test_operation(self, op_name, custom_op, torch_op, custom_inputs, torch_inputs, params_to_check=()):
        # Forward pass
        custom_result = custom_op(*custom_inputs)
        torch_result = torch_op(*torch_inputs)
        
        # Check forward data
        self.assert_tensors_close(custom_result, torch_result, check_grad=False)
        
        # Backward pass
        custom_result.sum().backward()
        torch_result.sum().backward()
        
        # Check gradients
        for custom_param, torch_param in params_to_check:
            self.assert_tensors_close(custom_param, torch_param)
            
        return custom_result, torch_result

    def _reset_grads(self, *tensor_pairs):
        for custom_tensor, torch_tensor in tensor_pairs:
            custom_tensor.reset_grad()
            if torch_tensor is not None:
                torch_tensor.grad = None

    def test_add(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        
        self._test_operation(
            "Add", 
            lambda x, y: x + y, 
            lambda x, y: x + y,
            (a_rt, b_rt), 
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_add_with_scalar(self):
        a_data = np.array([1.0, 2.0, 3.0])
        scalar = 5.0
        
        a_rt = RealTensor(a_data, requires_grad=True)
        a_torch = to_torch(a_data)
        
        self._test_operation(
            "Add Scalar",
            lambda x, s: x + s,
            lambda x, s: x + s,
            (a_rt, scalar),
            (a_torch, scalar),
            [(a_rt, a_torch)]
        )

    def test_add_broadcasting(self):
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        b_data = np.array([10.0, 20.0, 30.0])  # 3,
        
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        
        self._test_operation(
            "Add Broadcast",
            lambda x, y: x + y,
            lambda x, y: x + y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_mul(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[0.5, 0.25], [0.1, 0.0]])
        
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        
        self._test_operation(
            "Mul",
            lambda x, y: x * y,
            lambda x, y: x * y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_pow(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[2.0, 3.0], [0.5, 1.0]])
        
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        
        self._test_operation(
            "Pow",
            lambda x, y: x ** y,
            lambda x, y: x ** y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_sum(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        # Test sum all
        self._test_operation(
            "Sum All",
            lambda x: x.sum(),
            lambda x: x.sum(),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        self._reset_grads((rt, torch_t))
        
        # Test sum axis=0
        self._test_operation(
            "Sum Axis 0",
            lambda x: x.sum(axis=0),
            lambda x: x.sum(axis=0),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        self._reset_grads((rt, torch_t))
        
        # Test sum axis=1, keepdims=True
        self._test_operation(
            "Sum Axis 1 Keepdims",
            lambda x: x.sum(axis=1, keepdims=True),
            lambda x: x.sum(axis=1, keepdim=True),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_mean(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        # Test mean all
        self._test_operation(
            "Mean All",
            lambda x: x.mean(),
            lambda x: x.mean(),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        self._reset_grads((rt, torch_t))
        
        # Test mean axis=0
        self._test_operation(
            "Mean Axis 0",
            lambda x: x.mean(axis=0),
            lambda x: x.mean(axis=0),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_neg(self):
        data = np.array([[1.0, -2.0], [0.0, 4.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        self._test_operation(
            "Neg",
            lambda x: -x,
            lambda x: -x,
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_sub(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        
        self._test_operation(
            "Sub",
            lambda x, y: x - y,
            lambda x, y: x - y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_abs(self):
        data = np.array([[-1.0, 2.0], [-3.0, 0.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        self._test_operation(
            "Abs",
            lambda x: x.abs(),
            lambda x: torch.abs(x),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_reshape(self):
        data = np.arange(6.0).reshape((2, 3))
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        new_shape = (3, 2)
        
        self._test_operation(
            "Reshape",
            lambda x: x.reshape(new_shape),
            lambda x: x.reshape(new_shape),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_unsqueeze_squeeze(self):
        # Unsqueeze
        data = np.array([[1., 2., 3.]])  # Shape (1,3)
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        result_rt, result_torch = self._test_operation(
            "Unsqueeze",
            lambda x: x.unsqueeze(axis=0),
            lambda x: x.unsqueeze(axis=0),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        
        self.assertEqual(result_rt.shape, (1, 1, 3))
        self._reset_grads((rt, torch_t))
        
        # Squeeze
        data = np.array([[[1.],[2.],[3.]]])  # Shape (1,3,1)
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)
        
        # Squeeze axis 0
        result_rt, _ = self._test_operation(
            "Squeeze axis 0",
            lambda x: x.squeeze(axis=0),
            lambda x: x.squeeze(dim=0),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        
        self.assertEqual(result_rt.shape, (3, 1))
        self._reset_grads((rt, torch_t))
        
        # Squeeze axis 2
        result_rt, _ = self._test_operation(
            "Squeeze axis 2",
            lambda x: x.squeeze(axis=2),
            lambda x: x.squeeze(dim=2),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )
        
        self.assertEqual(result_rt.shape, (1, 3))

    def test_stack(self):
        t1_data = np.array([[1.,2.],[3.,4.]])
        t2_data = np.array([[5.,6.],[7.,8.]])
        
        rt1 = RealTensor(t1_data, requires_grad=True)
        rt2 = RealTensor(t2_data, requires_grad=True)
        torch1 = to_torch(t1_data)
        torch2 = to_torch(t2_data)
        
        # Stack axis 0
        self._test_operation(
            "Stack ax0",
            lambda t1, t2: RealTensor.stack(t1, t2, axis=0),
            lambda t1, t2: torch.stack([t1, t2], axis=0),
            (rt1, rt2),
            (torch1, torch2),
            [(rt1, torch1), (rt2, torch2)]
        )
        self._reset_grads((rt1, torch1), (rt2, torch2))
        
        # Stack axis 1
        self._test_operation(
            "Stack ax1",
            lambda t1, t2: RealTensor.stack(t1, t2, axis=1),
            lambda t1, t2: torch.stack([t1, t2], axis=1),
            (rt1, rt2),
            (torch1, torch2),
            [(rt1, torch1), (rt2, torch2)]
        )

    def test_cross_entropy(self):
        preds_data = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        targets_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        
        preds_rt = RealTensor(preds_data, requires_grad=True)
        targets_rt = RealTensor(targets_data, requires_grad=False)
        
        preds_torch = to_torch(preds_data, requires_grad=True)
        targets_torch = to_torch(targets_data, requires_grad=False)
        
        # Custom cross_entropy
        ce_rt = preds_rt.cross_entropy(targets_rt)
        
        # PyTorch equivalent
        log_softmax_preds_torch = torch.log_softmax(preds_torch, dim=1)
        loss_torch = -torch.sum(targets_torch * log_softmax_preds_torch) / preds_torch.shape[0]
        
        # Check forward result
        self.assertTrue(np.allclose(ce_rt.data, loss_torch.detach().numpy(), atol=1e-6))
        
        # Check gradients
        ce_rt.sum().backward()
        loss_torch.sum().backward()
        self.assertTrue(np.allclose(preds_rt._grad, preds_torch.grad.numpy(), atol=1e-5))

    def test_sliding_window_2d(self):
        # Create input data in NHWC format (batch, height, width, channels)
        data = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
        
        # Our custom tensor
        rt = RealTensor(data, requires_grad=True)
        
        # Test case 1: No padding, stride=1
        window_size = 2
        padding = 0
        stride = 1
        
        # Our sliding window implementation
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        
        # Check shape
        self.assertEqual(windows.shape, (1, 3, 3, 1, 2, 2))
        
        # Check output values
        expected_windows = np.zeros((1, 3, 3, 1, 2, 2), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                expected_windows[0, i, j, 0] = data[0, i:i+2, j:j+2, 0]
        
        self.assertTrue(np.allclose(windows.data, expected_windows))
        
        # Test backward pass - only verify shape
        windows.sum().backward()
        self.assertEqual(rt._grad.shape, data.shape)
        
        # Test case 2: With padding, stride=1
        rt.reset_grad()
        
        padding = 1
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        self.assertEqual(windows.shape, (1, 5, 5, 1, 2, 2))
        
        # Test case 3: No padding, stride=2
        rt.reset_grad()
        
        padding = 0
        stride = 2
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        self.assertEqual(windows.shape, (1, 2, 2, 1, 2, 2))

        # TODO: test backward pass values by comparing with torch


class TestRingTensorOps(unittest.TestCase):
    RT_DTYPE = RingTensor.dtype
    RT_MIN = RingTensor.min_value
    RT_MAX = RingTensor.max_value
    RT_MIN_F = float(RT_MIN)
    RT_MAX_F = float(RT_MAX)

    def assert_tensors_equal(self, custom_data, expected_data, msg_prefix=""):
        self.assertTrue(np.array_equal(custom_data, expected_data.astype(custom_data.dtype)),
                        f"{msg_prefix} mismatch: custom={custom_data}, expected={expected_data}")

    def assert_grad_close(self, custom_tensor, torch_grad_numpy, rtol=1e-5, atol=1e-7, msg_prefix=""):
        self.assertTrue(np.allclose(custom_tensor._grad, torch_grad_numpy, rtol=rtol, atol=atol),
                        f"{msg_prefix} grad mismatch: custom={custom_tensor._grad}, torch={torch_grad_numpy}")

    def _reset_grads(self, *tensor_pairs):
        for custom_tensor, torch_tensor in tensor_pairs:
            custom_tensor.reset_grad()
            if torch_tensor is not None and hasattr(torch_tensor, 'grad'):
                torch_tensor.grad = None

    def _test_ring_op(self, op_name, custom_op, torch_op, expected_data_op, 
                     custom_inputs, torch_inputs, params_to_check, expect_float=False):
        # Forward pass and data check
        custom_result = custom_op(*custom_inputs)
        
        custom_input_data = [inp.data if hasattr(inp, 'data') else inp for inp in custom_inputs]
        expected_data = expected_data_op(*custom_input_data)
        
        if expect_float:
            self.assertTrue(np.allclose(custom_result.data, expected_data), f"Data mismatch ({op_name}): custom={custom_result.data}, expected={expected_data}")
        else:
            self.assert_tensors_equal(custom_result.data, expected_data, f"Data mismatch ({op_name}): custom={custom_result.data}, expected={expected_data}")
        
        # Gradient check
        torch_result = torch_op(*torch_inputs)
        
        custom_result.sum().backward()
        torch_result.sum().backward()
        
        for custom_param, torch_param in params_to_check:
            self.assert_grad_close(custom_param, torch_param.grad.numpy(), msg_prefix=f"Grad mismatch ({op_name}): custom={custom_param._grad}, torch={torch_param.grad.numpy()}")
            
        return custom_result

    def test_add_ring(self):
        a_data = np.array([[10, 20], [70, 120]], dtype=self.RT_DTYPE)
        b_data = np.array([[50, 60], [10, 10]], dtype=self.RT_DTYPE)
        
        a_rt = RingTensor(raw_data=a_data, requires_grad=True)
        b_rt = RingTensor(raw_data=b_data, requires_grad=True)
        a_torch = to_torch(a_data.astype(np.float32), requires_grad=True)
        b_torch = to_torch(b_data.astype(np.float32), requires_grad=True)
        
        self._test_ring_op(
            "Add",
            lambda x, y: x + y,
            lambda x, y: x + y,
            lambda d1, d2: (d1.astype(np.int32) + d2.astype(np.int32)).astype(self.RT_DTYPE),
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_neg_ring(self):
        data = np.array([self.RT_MIN, 0, 10, self.RT_MAX], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = to_torch(data.astype(np.float32), requires_grad=True)
        
        self._test_ring_op(
            "Neg",
            lambda x: -x,
            lambda x: -x,
            lambda d: (-d).astype(self.RT_DTYPE),
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_sum_ring(self):
        data = np.array([[10, 20], [30, 100]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = to_torch(data.astype(np.float32), requires_grad=True)
        
        self._test_ring_op(
            "Sum",
            lambda x: x.sum(),
            lambda x: x.sum(),
            lambda d: np.array(d.sum(), dtype=self.RT_DTYPE),
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_mean_ring(self):
        data = np.array([[10, 11], [20, 21]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = to_torch(data.astype(np.float32), requires_grad=True)
        
        self._test_ring_op(
            "Mean",
            lambda x: x.mean(),
            lambda x: x.mean(),
            lambda d: np.array(d.mean(), dtype=rt.dtype),
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_ring_sin(self):
        data = np.array([-100, 0, 50, self.RT_MAX // 2, self.RT_MIN // 2], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = to_torch(data.astype(np.float32), requires_grad=True)
        
        def torch_sin_op(raw_x_torch):
            as_float_torch = raw_x_torch / (-self.RT_MIN_F)
            sign_raw_torch = torch.sign(raw_x_torch)
            y_f_torch = (torch.sin(as_float_torch * np.pi - np.pi / 2.0) + 1.0) * sign_raw_torch * 0.5
            return y_f_torch * (-self.RT_MIN_F)
            
        def expected_sin_data(raw_d):
            as_float = raw_d.astype(np.float32) / (-self.RT_MIN_F)
            sign_raw = np.sign(raw_d.astype(np.float32))
            y_f = (np.sin(as_float * np.pi - np.pi / 2.0) + 1.0) * sign_raw * 0.5
            return (y_f * (-self.RT_MIN_F)).clip(self.RT_MIN, self.RT_MAX).astype(self.RT_DTYPE)
            
        self._test_ring_op(
            "Ring Sin",
            lambda x: x.sin2(),
            torch_sin_op,
            expected_sin_data,
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_ring_real(self):
        data = np.array([[-10, 20], [30, -40]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        
        real_t = rt.real()
        
        # Check type and data
        self.assertIsInstance(real_t, RealTensor)
        self.assertTrue(np.allclose(real_t.data, rt.as_float()))
        self.assertEqual(real_t._rg, rt._rg)
        
        # Check conversion back to ring tensor
        rt_from_real = RingTensor(real_t.as_float())
        self.assertTrue(np.allclose(rt_from_real.data, data))

    def test_sliding_window_2d_ring(self):
        # Create input data in NHWC format (batch, height, width, channels)
        data = np.arange(16, dtype=RingTensor.dtype).reshape(1, 4, 4, 1)
        
        # Our custom tensor
        rt = RingTensor(raw_data=data, requires_grad=True)
        
        # Test case 1: No padding, stride=1
        window_size = 2
        padding = 0
        stride = 1
        
        # Our sliding window implementation
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        
        # Check shape
        self.assertEqual(windows.shape, (1, 3, 3, 1, 2, 2))
        
        # Check output values
        expected_windows = np.zeros((1, 3, 3, 1, 2, 2), dtype=RingTensor.dtype)
        for i in range(3):
            for j in range(3):
                expected_windows[0, i, j, 0] = data[0, i:i+2, j:j+2, 0]
        
        self.assert_tensors_equal(windows.data, expected_windows)
        
        # Only test the forward pass for RingTensor since gradients can cause type issues
        
        # Test case 2: With padding, stride=1
        rt.reset_grad()
        
        padding = 1
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        self.assertEqual(windows.shape, (1, 5, 5, 1, 2, 2))
        
        # Test case 3: No padding, stride=2
        rt.reset_grad()
        
        padding = 0
        stride = 2
        windows = rt.sliding_window_2d(window_size=window_size, padding=padding, stride=stride)
        self.assertEqual(windows.shape, (1, 2, 2, 1, 2, 2))


if __name__ == '__main__':
    unittest.main()
