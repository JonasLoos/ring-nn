import unittest
import torch
from tensor import RingTensor, RealTensor

class TestRealTensorOps(unittest.TestCase):
    def assert_tensors_close(self, custom_tensor, torch_tensor, rtol=1e-5, atol=1e-7, check_grad=True):
        # Check data values
        custom_data = custom_tensor.data.cpu().numpy() if hasattr(custom_tensor.data, 'cpu') else custom_tensor.data
        torch_data = torch_tensor.detach().cpu().numpy() if hasattr(torch_tensor, 'cpu') else torch_tensor.detach().numpy()
        self.assertTrue(torch.allclose(torch.tensor(custom_data), torch.tensor(torch_data), rtol=rtol, atol=atol),
                       f"Data mismatch: custom={custom_data}, torch={torch_data}")

        # Check gradients if needed
        if check_grad and custom_tensor._grad is not None and torch_tensor.grad is not None:
            custom_grad = custom_tensor._grad.cpu().numpy() if hasattr(custom_tensor._grad, 'cpu') else custom_tensor._grad
            torch_grad = torch_tensor.grad.cpu().numpy() if hasattr(torch_tensor.grad, 'cpu') else torch_tensor.grad.numpy()
            self.assertTrue(torch.allclose(torch.tensor(custom_grad), torch.tensor(torch_grad), rtol=rtol, atol=atol),
                           f"Grad mismatch: custom={custom_grad}, torch={torch_grad}")

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
        a_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_data = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)
        b_torch = b_data.clone().requires_grad_(True)

        self._test_operation(
            "Add",
            lambda x, y: x + y,
            lambda x, y: x + y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_add_with_scalar(self):
        a_data = torch.tensor([1.0, 2.0, 3.0])
        scalar = 5.0

        a_rt = RealTensor(a_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)

        self._test_operation(
            "Add Scalar",
            lambda x, s: x + s,
            lambda x, s: x + s,
            (a_rt, scalar),
            (a_torch, scalar),
            [(a_rt, a_torch)]
        )

    def test_add_broadcasting(self):
        a_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        b_data = torch.tensor([10.0, 20.0, 30.0])  # 3,

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)
        b_torch = b_data.clone().requires_grad_(True)

        self._test_operation(
            "Add Broadcast",
            lambda x, y: x + y,
            lambda x, y: x + y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_mul(self):
        a_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_data = torch.tensor([[0.5, 0.25], [0.1, 0.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)
        b_torch = b_data.clone().requires_grad_(True)

        self._test_operation(
            "Mul",
            lambda x, y: x * y,
            lambda x, y: x * y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_pow(self):
        a_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_data = torch.tensor([[2.0, 3.0], [0.5, 1.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)
        b_torch = b_data.clone().requires_grad_(True)

        self._test_operation(
            "Pow",
            lambda x, y: x ** y,
            lambda x, y: x ** y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_sum(self):
        data = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

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
        data = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

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
        data = torch.tensor([[1.0, -2.0], [0.0, 4.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

        self._test_operation(
            "Neg",
            lambda x: -x,
            lambda x: -x,
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_sub(self):
        a_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_data = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        a_torch = a_data.clone().requires_grad_(True)
        b_torch = b_data.clone().requires_grad_(True)

        self._test_operation(
            "Sub",
            lambda x, y: x - y,
            lambda x, y: x - y,
            (a_rt, b_rt),
            (a_torch, b_torch),
            [(a_rt, a_torch), (b_rt, b_torch)]
        )

    def test_abs(self):
        data = torch.tensor([[-1.0, 2.0], [-3.0, 0.0]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

        self._test_operation(
            "Abs",
            lambda x: x.abs(),
            lambda x: torch.abs(x),
            (rt,),
            (torch_t,),
            [(rt, torch_t)]
        )

    def test_reshape(self):
        data = torch.arange(6.0).reshape((2, 3))
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)
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
        data = torch.tensor([[1., 2., 3.]])  # Shape (1,3)
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

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
        data = torch.tensor([[[1.],[2.],[3.]]])  # Shape (1,3,1)
        rt = RealTensor(data, requires_grad=True)
        torch_t = data.clone().requires_grad_(True)

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
        t1_data = torch.tensor([[1.,2.],[3.,4.]])
        t2_data = torch.tensor([[5.,6.],[7.,8.]])

        rt1 = RealTensor(t1_data, requires_grad=True)
        rt2 = RealTensor(t2_data, requires_grad=True)
        torch1 = t1_data.clone().requires_grad_(True)
        torch2 = t2_data.clone().requires_grad_(True)

        # Stack axis 0
        self._test_operation(
            "Stack ax0",
            lambda t1, t2: RealTensor.stack(t1, t2, axis=0),
            lambda t1, t2: torch.stack([t1, t2], dim=0),
            (rt1, rt2),
            (torch1, torch2),
            [(rt1, torch1), (rt2, torch2)]
        )
        self._reset_grads((rt1, torch1), (rt2, torch2))

        # Stack axis 1
        self._test_operation(
            "Stack ax1",
            lambda t1, t2: RealTensor.stack(t1, t2, axis=1),
            lambda t1, t2: torch.stack([t1, t2], dim=1),
            (rt1, rt2),
            (torch1, torch2),
            [(rt1, torch1), (rt2, torch2)]
        )

    def test_cross_entropy(self):
        preds_data = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        targets_data = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        preds_rt = RealTensor(preds_data, requires_grad=True)
        targets_rt = RealTensor(targets_data, requires_grad=False)

        preds_torch = preds_data.clone().requires_grad_(True)
        targets_torch = targets_data.clone().requires_grad_(False)

        # Custom cross_entropy
        ce_rt = preds_rt.cross_entropy(targets_rt)

        # PyTorch equivalent
        log_softmax_preds_torch = torch.log_softmax(preds_torch, dim=1)
        loss_torch = -torch.sum(targets_torch * log_softmax_preds_torch) / preds_torch.shape[0]

        # Check forward result
        ce_rt_np = ce_rt.data.cpu().numpy() if hasattr(ce_rt.data, 'cpu') else ce_rt.data
        loss_torch_np = loss_torch.detach().cpu().numpy() if hasattr(loss_torch, 'cpu') else loss_torch.detach().numpy()
        self.assertTrue(torch.allclose(torch.tensor(ce_rt_np), torch.tensor(loss_torch_np), atol=1e-6))

        # Check gradients
        ce_rt.sum().backward()
        loss_torch.sum().backward()
        assert preds_rt._grad is not None
        assert preds_torch.grad is not None
        preds_rt_grad_np = preds_rt._grad.cpu().numpy() if hasattr(preds_rt._grad, 'cpu') else preds_rt._grad
        preds_torch_grad_np = preds_torch.grad.cpu().numpy() if hasattr(preds_torch.grad, 'cpu') else preds_torch.grad.numpy()
        self.assertTrue(torch.allclose(torch.tensor(preds_rt_grad_np), torch.tensor(preds_torch_grad_np), atol=1e-5))

    def test_sliding_window_2d(self):
        # Create input data in NHWC format (batch, height, width, channels)
        data = torch.arange(16).reshape(1, 4, 4, 1).float()

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
        expected_windows = torch.zeros((1, 3, 3, 1, 2, 2))
        for i in range(3):
            for j in range(3):
                expected_windows[0, i, j, 0] = data[0, i:i+2, j:j+2, 0]

        windows_np = windows.data.cpu().numpy() if hasattr(windows.data, 'cpu') else windows.data
        expected_windows_np = expected_windows.cpu().numpy() if hasattr(expected_windows, 'cpu') else expected_windows
        self.assertTrue(torch.allclose(torch.tensor(windows_np), torch.tensor(expected_windows_np)))

        # Test backward pass - only verify shape
        windows.sum().backward()
        assert rt._grad is not None
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
    RT_MIN = int(RingTensor.min_value)
    RT_MAX = int(RingTensor.max_value)
    RT_MIN_F = float(RT_MIN)
    RT_MAX_F = float(RT_MAX)

    def assert_tensors_equal(self, custom_data, expected_data, msg_prefix=""):
        custom_np = custom_data.cpu().numpy() if hasattr(custom_data, 'cpu') else custom_data
        expected_np = expected_data.cpu().numpy() if hasattr(expected_data, 'cpu') else expected_data
        self.assertTrue(torch.allclose(torch.tensor(custom_np), torch.tensor(expected_np)),
                        f"{msg_prefix} mismatch: custom={custom_np}, expected={expected_np}")

    def assert_grad_close(self, custom_tensor, torch_grad_numpy, rtol=1e-5, atol=1e-7, msg_prefix=""):
        custom_grad = custom_tensor._grad.cpu().numpy() if hasattr(custom_tensor._grad, 'cpu') else custom_tensor._grad
        self.assertTrue(torch.allclose(torch.tensor(custom_grad), torch.tensor(torch_grad_numpy), rtol=rtol, atol=atol),
                        f"{msg_prefix} grad mismatch: custom={custom_grad}, torch={torch_grad_numpy}")

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

        custom_data_np = custom_result.data.cpu().numpy() if hasattr(custom_result.data, 'cpu') else custom_result.data
        if expect_float:
            self.assertTrue(torch.allclose(torch.tensor(custom_data_np), torch.tensor(expected_data)), f"Data mismatch ({op_name}): custom={custom_data_np}, expected={expected_data}")
        else:
            self.assert_tensors_equal(custom_result.data, expected_data, f"Data mismatch ({op_name}): custom={custom_data_np}, expected={expected_data}")

        # Gradient check
        torch_result = torch_op(*torch_inputs)

        custom_result.sum().backward()
        torch_result.sum().backward()

        for custom_param, torch_param in params_to_check:
            self.assert_grad_close(custom_param, torch_param.grad.numpy(), msg_prefix=f"Grad mismatch ({op_name}): custom={custom_param._grad}, torch={torch_param.grad.numpy()}")

        return custom_result

    def test_add_ring(self):
        a_data = torch.tensor([[10, 20], [70, 120]], dtype=self.RT_DTYPE)
        b_data = torch.tensor([[50, 60], [10, 10]], dtype=self.RT_DTYPE)

        a_rt = RingTensor(raw_data=a_data, requires_grad=True)
        b_rt = RingTensor(raw_data=b_data, requires_grad=True)

        # For gradient computation, we need float tensors
        a_torch_float = a_data.float().requires_grad_(True)
        b_torch_float = b_data.float().requires_grad_(True)

        def expected_add(d1, d2):
            # d1 and d2 are torch tensors, convert to numpy for the operation
            d1_np = d1.cpu().numpy() if hasattr(d1, 'cpu') else d1
            d2_np = d2.cpu().numpy() if hasattr(d2, 'cpu') else d2
            result = torch.tensor(d1_np, dtype=torch.int32) + torch.tensor(d2_np, dtype=torch.int32)
            return result.to(self.RT_DTYPE)

        self._test_ring_op(
            "Add",
            lambda x, y: x + y,
            lambda x, y: x + y,
            expected_add,
            (a_rt, b_rt),
            (a_torch_float, b_torch_float),
            [(a_rt, a_torch_float), (b_rt, b_torch_float)]
        )

    def test_neg_ring(self):
        data = torch.tensor([self.RT_MIN, 0, 10, self.RT_MAX], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = data.float().requires_grad_(True)

        def expected_neg(d):
            d_np = d.cpu().numpy() if hasattr(d, 'cpu') else d
            return torch.tensor(-d_np, dtype=self.RT_DTYPE)

        self._test_ring_op(
            "Neg",
            lambda x: -x,
            lambda x: -x,
            expected_neg,
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_sum_ring(self):
        data = torch.tensor([[10, 20], [30, 100]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = data.float().requires_grad_(True)

        def expected_sum(d):
            d_np = d.cpu().numpy() if hasattr(d, 'cpu') else d
            return torch.tensor(d_np.sum(), dtype=self.RT_DTYPE)

        self._test_ring_op(
            "Sum",
            lambda x: x.sum(),
            lambda x: x.sum(),
            expected_sum,
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_mean_ring(self):
        data = torch.tensor([[10, 11], [20, 21]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = data.float().requires_grad_(True)

        def expected_mean(d):
            d_np = d.cpu().numpy() if hasattr(d, 'cpu') else d
            return torch.tensor(d_np.mean(), dtype=self.RT_DTYPE)

        self._test_ring_op(
            "Mean",
            lambda x: x.mean(),
            lambda x: x.mean(),
            expected_mean,
            (rt,),
            (torch_data,),
            [(rt, torch_data)]
        )

    def test_ring_sin(self):
        data = torch.tensor([-100, 0, 50, self.RT_MAX // 2, self.RT_MIN // 2], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)
        torch_data = data.float().requires_grad_(True)

        def torch_sin_op(raw_x_torch):
            as_float_torch = raw_x_torch / (-self.RT_MIN_F)
            sign_raw_torch = torch.sign(raw_x_torch)
            y_f_torch = (torch.sin(as_float_torch * torch.pi - torch.pi / 2.0) + 1.0) * sign_raw_torch * 0.5
            return y_f_torch * (-self.RT_MIN_F)

        def expected_sin_data(raw_d):
            raw_d_np = raw_d.cpu().numpy() if hasattr(raw_d, 'cpu') else raw_d
            as_float = torch.tensor(raw_d_np, dtype=torch.float32) / (-self.RT_MIN_F)
            sign_raw = torch.sign(torch.tensor(raw_d_np, dtype=torch.float32))
            y_f = (torch.sin(as_float * torch.pi - torch.pi / 2.0) + 1.0) * sign_raw * 0.5
            result = y_f * (-self.RT_MIN_F)
            return torch.clamp(result, self.RT_MIN, self.RT_MAX).to(torch.int16)

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
        data = torch.tensor([[-10, 20], [30, -40]], dtype=self.RT_DTYPE)
        rt = RingTensor(raw_data=data, requires_grad=True)

        real_t = rt.real()

        # Check type and data
        self.assertIsInstance(real_t, RealTensor)
        real_data_np = real_t.data.cpu().numpy() if hasattr(real_t.data, 'cpu') else real_t.data
        rt_float_np = rt.as_float().cpu().numpy() if hasattr(rt.as_float(), 'cpu') else rt.as_float()
        self.assertTrue(torch.allclose(torch.tensor(real_data_np), torch.tensor(rt_float_np)))
        self.assertEqual(real_t._rg, rt._rg)

        # Check conversion back to ring tensor
        rt_from_real = RingTensor(real_t.as_float())
        rt_from_real_np = rt_from_real.data.cpu().numpy() if hasattr(rt_from_real.data, 'cpu') else rt_from_real.data
        data_np = data.cpu().numpy() if hasattr(data, 'cpu') else data
        self.assertTrue(torch.allclose(torch.tensor(rt_from_real_np), torch.tensor(data_np)))

    def test_sliding_window_2d_ring(self):
        # Create input data in NHWC format (batch, height, width, channels)
        data = torch.arange(16, dtype=self.RT_DTYPE).reshape(1, 4, 4, 1)

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
        expected_windows = torch.zeros((1, 3, 3, 1, 2, 2), dtype=self.RT_DTYPE)
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
