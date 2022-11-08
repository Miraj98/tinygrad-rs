use ndarray::arr2;
use tinygrad_rust::tensor::ops::{Backprop, BinaryOps, TensorConstructors};
use tinygrad_rust::tensor::TensorCore;

fn main() {
    let a = TensorCore::new(arr2(&[[4., 3.], [2., 1.]]));
    let b = TensorCore::new(arr2(&[[1., 2.], [3., 4.]]));
    let c = a.matmul(&b);

    c.backward();

    let b_grad = b.data.grad.borrow();
    println!("b_grad: {:?}", b_grad);

    let a_grad = a.data.grad.borrow();
    println!("a_grad: {:?}", a_grad);

    let c_grad = c.data.grad.borrow();
    println!("c_grad: {:?}", c_grad);
}
