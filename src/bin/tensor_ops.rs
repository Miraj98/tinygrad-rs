use ndarray::arr2;
use tinygrad_rust::tensor::ops::{Backprop, BinaryOps, TensorConstructors, UnaryOps, ReduceOps};
use tinygrad_rust::tensor::TensorCore;

fn main() {
    let a = TensorCore::new(arr2(&[[4., 3.], [2., 1.]]), true);
    let b = TensorCore::new(arr2(&[[1., 2.], [3., 4.]]), true);
    // let c = a.matmul(&b).add(x);

    let loss = a
    .sub(&b)
    .mul(&TensorCore::fill(
        a.data.value.borrow().dim(),
        0.5,
        true
    ))
    .square()
    .mean();

    loss.backward();

    println!("{:#?}", loss);

    let b_grad = b.data.grad.borrow();
    println!("b_grad: {:?}", b_grad);

    let a_grad = a.data.grad.borrow();
    println!("a_grad: {:?}", a_grad);

    let loss_grad = loss.data.grad.borrow();
    println!("c_grad: {:?}", loss_grad);
}
