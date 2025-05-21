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

    def test_add(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        c_rt = a_rt + b_rt

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        c_torch = a_torch + b_torch

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Add Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Add Op (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Add Op (Grad b): ")

    def test_add_with_scalar(self):
        a_data = np.array([1.0, 2.0, 3.0])
        scalar = 5.0
        a_rt = RealTensor(a_data, requires_grad=True)
        c_rt = a_rt + scalar

        a_torch = to_torch(a_data)
        c_torch = a_torch + scalar

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Add Scalar Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Add Scalar Op (Grad a): ")

    def test_add_broadcasting(self):
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # 2x3
        b_data = np.array([10.0, 20.0, 30.0]) # 3,

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        c_rt = a_rt + b_rt

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        c_torch = a_torch + b_torch

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Add Broadcast Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Add Broadcast Op (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Add Broadcast Op (Grad b): ")


    def test_mul(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[0.5, 0.25], [0.1, 0.0]])
        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        c_rt = a_rt * b_rt

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        c_torch = a_torch * b_torch

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Mul Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Mul Op (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Mul Op (Grad b): ")

    def test_pow(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[2.0, 3.0], [0.5, 1.0]]) # Exponent
        a_rt = RealTensor(a_data, requires_grad=True)
        # Exponent does not require grad in torch if it's not a tensor, or if it is, it also gets grad.
        # For simplicity, make exponent RealTensor too, matching custom Tensor behavior.
        b_rt = RealTensor(b_data, requires_grad=True)
        c_rt = a_rt ** b_rt

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data) # Exponent as tensor
        c_torch = a_torch ** b_torch

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Pow Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Pow Op (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Pow Op (Grad b): ")


    def test_sum(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)

        # Test sum all
        sum_rt_all = rt.sum()
        sum_torch_all = torch_t.sum()
        self._assert_tensor_data_close(sum_rt_all.data, sum_torch_all.detach().numpy(), msg_prefix="Sum All Op: ")
        sum_rt_all.backward() # Already scalar
        sum_torch_all.backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Sum All Op (Grad): ")

        rt.reset_grad(); torch_t.grad = None # Reset for next test

        # Test sum axis=0
        sum_rt_axis0 = rt.sum(axis=0)
        sum_torch_axis0 = torch_t.sum(axis=0)
        self._assert_tensor_data_close(sum_rt_axis0.data, sum_torch_axis0.detach().numpy(), msg_prefix="Sum Axis 0 Op: ")
        sum_rt_axis0.sum().backward() # Sum to scalar for backward
        sum_torch_axis0.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Sum Axis 0 Op (Grad): ")

        rt.reset_grad(); torch_t.grad = None

        # Test sum axis=1, keepdims=True
        sum_rt_axis1_keepdims = rt.sum(axis=1, keepdims=True)
        sum_torch_axis1_keepdims = torch_t.sum(axis=1, keepdim=True)
        self._assert_tensor_data_close(sum_rt_axis1_keepdims.data, sum_torch_axis1_keepdims.detach().numpy(), msg_prefix="Sum Axis 1 Keepdims Op: ")
        sum_rt_axis1_keepdims.sum().backward()
        sum_torch_axis1_keepdims.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Sum Axis 1 Keepdims Op (Grad): ")

    def test_mean(self):
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)

        # Test mean all
        mean_rt_all = rt.mean()
        mean_torch_all = torch_t.mean()
        self._assert_tensor_data_close(mean_rt_all.data, mean_torch_all.detach().numpy(), msg_prefix="Mean All Op: ")
        mean_rt_all.backward()
        mean_torch_all.backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Mean All Op (Grad): ")

        rt.reset_grad(); torch_t.grad = None

        # Test mean axis=0
        mean_rt_axis0 = rt.mean(axis=0)
        mean_torch_axis0 = torch_t.mean(axis=0)
        self._assert_tensor_data_close(mean_rt_axis0.data, mean_torch_axis0.detach().numpy(), msg_prefix="Mean Axis 0 Op: ")
        mean_rt_axis0.sum().backward()
        mean_torch_axis0.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Mean Axis 0 Op (Grad): ")

    def test_neg(self):
        data = np.array([[1.0, -2.0], [0.0, 4.0]])
        rt = RealTensor(data, requires_grad=True)
        neg_rt = -rt

        torch_t = to_torch(data)
        neg_torch = -torch_t

        self._assert_tensor_data_close(neg_rt.data, neg_torch.detach().numpy(), msg_prefix="Neg Op: ")
        neg_rt.sum().backward()
        neg_torch.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Neg Op (Grad): ")

    def test_sub(self):
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        a_rt = RealTensor(a_data, requires_grad=True)
        b_rt = RealTensor(b_data, requires_grad=True)
        c_rt = a_rt - b_rt

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        c_torch = a_torch - b_torch

        self._assert_tensor_data_close(c_rt.data, c_torch.detach().numpy(), msg_prefix="Sub Op: ")
        c_rt.sum().backward()
        c_torch.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Sub Op (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Sub Op (Grad b): ")

    def test_abs(self):
        data = np.array([[-1.0, 2.0], [-3.0, 0.0]])
        rt = RealTensor(data, requires_grad=True)
        abs_rt = rt.abs()

        torch_t = to_torch(data)
        abs_torch = torch.abs(torch_t)

        self._assert_tensor_data_close(abs_rt.data, abs_torch.detach().numpy(), msg_prefix="Abs Op: ")
        abs_rt.sum().backward()
        abs_torch.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Abs Op (Grad): ")

    def test_reshape(self):
        data = np.arange(6.0).reshape((2, 3))
        rt = RealTensor(data, requires_grad=True)
        reshaped_rt = rt.reshape((3, 2))

        torch_t = to_torch(data)
        reshaped_torch = torch_t.reshape((3, 2))

        self._assert_tensor_data_close(reshaped_rt.data, reshaped_torch.detach().numpy(), msg_prefix="Reshape Op: ")
        reshaped_rt.sum().backward()
        # To backprop through reshape in torch, the original tensor needs to be modified or a new one created that tracks grads.
        # Here, reshaped_torch already tracks grads from torch_t.
        reshaped_torch.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Reshape Op (Grad): ")

    def test_unsqueeze_squeeze(self):
        data = np.array([[1., 2., 3.]]) # Shape (1,3)
        rt = RealTensor(data, requires_grad=True)
        torch_t = to_torch(data)

        # Unsqueeze
        unsqueezed_rt = rt.unsqueeze(axis=0) # -> (1,1,3)
        unsqueezed_torch = torch_t.unsqueeze(axis=0)
        self.assertEqual(unsqueezed_rt.shape, (1,1,3))
        self._assert_tensor_data_close(unsqueezed_rt.data, unsqueezed_torch.detach().numpy(), msg_prefix="Unsqueeze Op: ")

        unsqueezed_rt.sum().backward()
        unsqueezed_torch.sum().backward()
        self._assert_tensor_grad_close(rt, torch_t, msg_prefix="Unsqueeze Op (Grad): ")

        rt.reset_grad(); torch_t.grad = None

        # Squeeze
        # data for squeeze: shape (1,3,1)
        data_s = np.array([[[1.],[2.],[3.]]])
        rt_s = RealTensor(data_s, requires_grad=True)
        torch_s = to_torch(data_s)

        squeezed_rt = rt_s.squeeze(axis=0) # -> (3,1)
        squeezed_torch = torch_s.squeeze(dim=0)
        self.assertEqual(squeezed_rt.shape, (3,1))
        self._assert_tensor_data_close(squeezed_rt.data, squeezed_torch.detach().numpy(), msg_prefix="Squeeze Op (axis=0): ")

        squeezed_rt.sum().backward()
        squeezed_torch.sum().backward()
        self._assert_tensor_grad_close(rt_s, torch_s, msg_prefix="Squeeze Op (axis=0) (Grad): ")

        rt_s.reset_grad(); torch_s.grad = None

        squeezed_rt_ax2 = rt_s.squeeze(axis=2) # -> (1,3)
        squeezed_torch_ax2 = torch_s.squeeze(dim=2)
        self.assertEqual(squeezed_rt_ax2.shape, (1,3))
        self._assert_tensor_data_close(squeezed_rt_ax2.data, squeezed_torch_ax2.detach().numpy(), msg_prefix="Squeeze Op (axis=2): ")

        squeezed_rt_ax2.sum().backward()
        squeezed_torch_ax2.sum().backward()
        self._assert_tensor_grad_close(rt_s, torch_s, msg_prefix="Squeeze Op (axis=2) (Grad): ")


    def test_stack(self):
        t1_data = np.array([[1.,2.],[3.,4.]])
        t2_data = np.array([[5.,6.],[7.,8.]])

        rt1 = RealTensor(t1_data, requires_grad=True)
        rt2 = RealTensor(t2_data, requires_grad=True)

        torch1 = to_torch(t1_data)
        torch2 = to_torch(t2_data)

        # Stack axis 0
        stacked_rt_ax0 = RealTensor.stack(rt1, rt2, axis=0)
        stacked_torch_ax0 = torch.stack([torch1, torch2], axis=0)
        self._assert_tensor_data_close(stacked_rt_ax0.data, stacked_torch_ax0.detach().numpy(), msg_prefix="Stack ax0 Op: ")

        stacked_rt_ax0.sum().backward()
        stacked_torch_ax0.sum().backward()
        self._assert_tensor_grad_close(rt1, torch1, msg_prefix="Stack ax0 Op (Grad rt1): ")
        self._assert_tensor_grad_close(rt2, torch2, msg_prefix="Stack ax0 Op (Grad rt2): ")

        rt1.reset_grad(); rt2.reset_grad()
        torch1.grad = None; torch2.grad = None

        # Stack axis 1
        stacked_rt_ax1 = RealTensor.stack(rt1, rt2, axis=1)
        stacked_torch_ax1 = torch.stack([torch1, torch2], axis=1)
        self._assert_tensor_data_close(stacked_rt_ax1.data, stacked_torch_ax1.detach().numpy(), msg_prefix="Stack ax1 Op: ")

        stacked_rt_ax1.sum().backward()
        stacked_torch_ax1.sum().backward()
        self._assert_tensor_grad_close(rt1, torch1, msg_prefix="Stack ax1 Op (Grad rt1): ")
        self._assert_tensor_grad_close(rt2, torch2, msg_prefix="Stack ax1 Op (Grad rt2): ")

    def test_cross_entropy(self):
        preds_data = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]) # (N,C) = (2,3)
        targets_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) # (N,C)

        preds_rt = RealTensor(preds_data, requires_grad=True)
        targets_rt = RealTensor(targets_data, requires_grad=False) # Typically targets don't require grad

        preds_torch = to_torch(preds_data, requires_grad=True)
        targets_torch = to_torch(targets_data, requires_grad=False)

        ce_rt_axis0 = preds_rt.cross_entropy(targets_rt)
        loss_torch_axis0 = torch.sum(-targets_torch * torch.log(preds_torch + 1e-9), dim=0)

        self._assert_tensor_data_close(ce_rt_axis0.data, loss_torch_axis0.detach().numpy(), msg_prefix="CE ax0 Op: ", atol=1e-6)

        ce_rt_axis0.sum().backward() # Sum to scalar
        loss_torch_axis0.sum().backward()

        self._assert_tensor_grad_close(preds_rt, preds_torch, msg_prefix="CE ax0 Op (Grad preds): ", atol=1e-5)

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

    def test_add_ring(self):
        a_data = np.array([[10, 20], [70, 120]], dtype=self.RT_DTYPE) # 120+10=130 -> -126 if int8
        b_data = np.array([[50, 60], [10, 10]], dtype=self.RT_DTYPE)

        a_rt = RingTensor(a_data, requires_grad=True)
        b_rt = RingTensor(b_data, requires_grad=True)
        c_rt = a_rt + b_rt

        expected_c_data = (a_data.astype(np.int32) + b_data.astype(np.int32)).astype(self.RT_DTYPE) # Simulate numpy's int arithmetic and cast
        self._assert_ring_tensor_data_equal(c_rt.data, expected_c_data, msg_prefix="Ring Add Data: ")

        a_torch = to_torch(a_data)
        b_torch = to_torch(b_data)
        c_torch_float = a_torch + b_torch # Grads based on float operation

        c_rt.sum().backward()
        c_torch_float.sum().backward()
        self._assert_tensor_grad_close(a_rt, a_torch, msg_prefix="Ring Add (Grad a): ")
        self._assert_tensor_grad_close(b_rt, b_torch, msg_prefix="Ring Add (Grad b): ")

    def test_neg_ring(self):
        data_vals = np.array([self.RT_MIN, np.iinfo(self.RT_DTYPE).min, 0, 10, self.RT_MAX, np.iinfo(self.RT_DTYPE).min + 1], dtype=self.RT_DTYPE)

        rt = RingTensor(data_vals, requires_grad=True)
        neg_rt = -rt

        expected_neg_data = -data_vals.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(neg_rt.data, expected_neg_data, msg_prefix="Ring Neg Data: ")

        # For grad, use float equivalent of the operation:
        data_torch = to_torch(data_vals)
        neg_torch_float = -data_torch

        neg_rt.sum().backward()
        neg_torch_float.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Neg (Grad): ")


    def test_sum_ring(self):
        data = np.array([[10, 20], [30, 100]], dtype=self.RT_DTYPE) # 10+20+30+100 = 160. For int8, 160 -> -96
        rt = RingTensor(data, requires_grad=True)

        # Forward
        sum_rt = rt.sum()
        # Numpy's sum on int8 might upcast (e.g. to int64), then RingTensor constructor casts back to RT_DTYPE
        expected_sum_data = np.array(data.sum(), dtype=self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(sum_rt.data, expected_sum_data, msg_prefix="Ring Sum Data: ")

        # Grad
        torch_data = to_torch(data)
        torch_sum_float = torch_data.sum()

        sum_rt.backward() # sum_rt is scalar
        torch_sum_float.backward()
        self._assert_tensor_grad_close(rt, torch_data, msg_prefix="Ring Sum (Grad): ")

    def test_mean_ring(self):
        data = np.array([[10, 11], [20, 21]], dtype=self.RT_DTYPE) # Mean = (10+11+20+21)/4 = 62/4 = 15.5
        rt = RingTensor(data, requires_grad=True)

        # Forward
        mean_rt = rt.mean()
        # Numpy's mean on int8 gives float64. RingTensor constructor casts to RT_DTYPE (e.g. int8(15.5) -> 15)
        expected_mean_data = np.array(data.mean(), dtype=self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(mean_rt.data, expected_mean_data, msg_prefix="Ring Mean Data: ")

        # Grad
        torch_data = to_torch(data)
        torch_mean_float = torch_data.mean()

        mean_rt.backward()
        torch_mean_float.backward()
        self._assert_tensor_grad_close(rt, torch_data, msg_prefix="Ring Mean (Grad): ")


    def test_ring_square(self):
        # (self.data.astype(np.float32) / self.min_value)**2 * -self.min_value * np.sign(self.data)
        # Then RingTensor constructor casts this to RingTensor.dtype (int8)
        # min_value in class is -127 for int8.
        data_val = np.array([-64, 0, 32, self.RT_MIN, self.RT_MAX], dtype=self.RT_DTYPE)
        rt = RingTensor(data_val, requires_grad=True)
        squared_rt = rt.square()

        # Expected forward data:
        float_data = data_val.astype(np.float32)
        # Note: RingTensor.min_value might be -127. If data_val has -128, np.sign(-128) is -1.
        # Use defined self.RT_MIN
        calc_float = (float_data / self.RT_MIN)**2 * (-self.RT_MIN) * np.sign(float_data)
        expected_data_int = calc_float.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(squared_rt.data, expected_data_int, msg_prefix="Ring Square Data: ")

        # PyTorch equivalent for grad
        data_torch = to_torch(data_val)
        min_val_torch = torch.tensor(float(self.RT_MIN))

        # Calculation in float for grad
        x_norm_torch = data_torch / min_val_torch
        sign_data_torch = torch.sign(data_torch)
        # Ensure sign(0) = 0 to match np.sign(0)=0
        sign_data_torch[data_torch == 0] = 0.0

        result_float_torch = (x_norm_torch**2) * (-min_val_torch) * sign_data_torch

        squared_rt.sum().backward()
        result_float_torch.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Square (Grad): ", atol=1e-6)


    def test_ring_sin(self):
        # activation = ((np.sin(x*np.pi - np.pi/2) + 1) * np.sign(x) * self.max_value).clip(self.min_value, self.max_value)
        # x = self.data.astype(np.float32) / (self.max_value - self.min_value)
        data_val = np.array([-100, 0, 50, self.RT_MAX, self.RT_MIN], dtype=self.RT_DTYPE)
        rt = RingTensor(data_val, requires_grad=True)
        sin_rt = rt.sin()

        # Expected forward data
        float_data = data_val.astype(np.float32)
        # max_value = 127, min_value = -127 for int8 example
        # Denominator (max-min) should not be zero. (127 - (-127)) = 254
        # If max_value == min_value (e.g. if range is just one number, unlikely for this class def), this would div by zero.
        # Assume max_value > min_value.
        denom = float(self.RT_MAX - self.RT_MIN)
        if denom == 0: denom = 1e-9 # Avoid division by zero if max=min

        x_norm = float_data / denom

        # np.sign(x) in original means np.sign(x_norm)
        act_float = (np.sin(x_norm * np.pi - np.pi/2) + 1) * np.sign(x_norm) * self.RT_MAX
        act_clipped_float = act_float.clip(self.RT_MIN, self.RT_MAX)
        expected_data_int = act_clipped_float.astype(self.RT_DTYPE)
        self._assert_ring_tensor_data_equal(sin_rt.data, expected_data_int, msg_prefix="Ring Sin Data: ")

        # PyTorch for grad
        data_torch = to_torch(data_val)
        min_val_f = float(self.RT_MIN)
        max_val_f = float(self.RT_MAX)
        denom_torch = torch.tensor(max_val_f - min_val_f if (max_val_f - min_val_f) !=0 else 1e-9)

        x_normalized_torch = data_torch / denom_torch
        sign_x_norm_torch = torch.sign(x_normalized_torch)
        sign_x_norm_torch[x_normalized_torch == 0] = 0.0 # Match np.sign(0)=0 behavior if it matters

        val_for_sin_torch = x_normalized_torch * np.pi - np.pi/2
        act_torch_f = (torch.sin(val_for_sin_torch) + 1) * sign_x_norm_torch * max_val_f
        act_torch_clipped = act_torch_f.clip(min_val_f, max_val_f)

        sin_rt.sum().backward()
        act_torch_clipped.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring Sin (Grad): ", atol=1e-5) # Higher atol due to trig/pi

    def test_ring_softmin(self):
        # axis=0 is the default in method sig, but not used in code? Let's test with axis=0.
        # softmin(self, axis=0)
        # abs_x = np.abs(self.data.astype(np.float32))
        # S     = abs_x.sum(axis, keepdims=True)
        # x_exp = np.exp(-abs_x / (S + 1e-9)) # Added epsilon for test stability
        # y     = x_exp / (x_exp.sum(axis, keepdims=True) + 1e-9)
        # out_val = (y * self.max_value).clip(self.min_value, self.max_value)

        data_val = np.array([[10, -20], [30, -40]], dtype=self.RT_DTYPE)
        axis_param = 0
        rt = RingTensor(data_val, requires_grad=True)
        softmin_rt = rt.softmin(axis=axis_param)

        # Expected forward pass
        float_data = data_val.astype(np.float32)
        abs_x = np.abs(float_data)
        S = abs_x.sum(axis=axis_param, keepdims=True)
        # Add epsilon for safety, though original code doesn't explicitly
        S_safe = S + 1e-9 if np.any(S==0) else S

        x_exp = np.exp(-abs_x / S_safe)
        y_denom = x_exp.sum(axis=axis_param, keepdims=True)
        y_denom_safe = y_denom + 1e-9 if np.any(y_denom==0) else y_denom
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
        S_torch_safe = S_torch + 1e-9 # Add epsilon for stability matching manual calc

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
        real_t = rt.real()

        self.assertIsInstance(real_t, RealTensor)
        self.assertTrue(np.array_equal(real_t.data, data_val.astype(np.float32)))
        self.assertEqual(real_t._rg, rt._rg)

        data_torch = to_torch(data_val)
        # Real conversion in torch is just ensuring float type, which to_torch does. Grad pass-through.
        real_torch = data_torch.clone()

        real_t.sum().backward()
        real_torch.sum().backward()
        self._assert_tensor_grad_close(rt, data_torch, msg_prefix="Ring.real (Grad):")

if __name__ == '__main__':
    unittest.main()