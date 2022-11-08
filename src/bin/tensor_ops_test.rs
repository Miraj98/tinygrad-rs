use ndarray::arr2;
use tinygrad_rust::tensor::ops::{TensorConstructors, BinaryOps, Backprop};
use tinygrad_rust::tensor::{TensorCore};

fn main() {
    let a = TensorCore::new(arr2(&[[4., 3.], [2., 1.]]));
    let b = TensorCore::new(arr2(&[[1., 2.], [3., 4.]]));

    let c = a.mul(&b).add(&a);

    c.backward();

    // let a = Tensor::new(arr2(&[
    //     [4., 3.],
    //     [2., 1.]
    // ]));
    // let b = Tensor::new(arr2(&[
    //     [1., 2.],
    //     [3., 4.]
    // ]));
    //     let c = a.matmul(&b);

    //     c.backward();

    //     let b_grad = b.grad.borrow();
    //     println!("b_grad: {:?}", b_grad);

    //     let a_grad = a.grad.borrow();
    //     println!("a_grad: {:?}", a_grad);

    //     let c_grad = c.grad.borrow();
    //     println!("c_grad: {:?}", c_grad);
}
