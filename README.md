### Example

```rust
use tinygrad_rust::tensor::{Tensor, ops::binary_ops::BinaryOps, Tensor};

fn main() {
  let x = Tensor::randn((2, 2), Some(true)); // requires_grad = Some(true)
  let y = Tensor::randn((2, 2), Some(true));
  let z = y.matmul(&x).sum();
  z.backward();
  
  println!("{:#?}", x.grad().as_ref().unwrap()); // dz/dx
  println!("{:#?}", y.grad().as_ref().unwrap()); // dz/dy
}
```

### Same example in torch

```python
import torch
x = torch.rand(2, 2, requires_grad=True)
y = torch.rand(2, 2, requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```
