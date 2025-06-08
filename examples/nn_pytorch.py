# Forward pass (shapes tracked automatically)
a = torch.randn(2, 3, requires_grad=True)  # Shape: (2,3)
b = torch.randn(3, 4, requires_grad=True)  # Shape: (3,4)
c = a @ b  # Shape: (2,4) [Node: MmBackward]
loss = c.sum()  # Scalar [Node: SumBackward]

# Backward pass (gradients match original shapes)
loss.backward()
print(a.grad.shape)  # (2,3) - matches 'a'
print(b.grad.shape)  # (3,4) - matches 'b'
