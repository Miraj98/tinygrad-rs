use tinygrad_rust::tensor::Tensor;
use ndarray::arr2;

fn main() {
    let a = Tensor::new(arr2(&[[3.]]));
    let b = Tensor::new(arr2(&[[12.]]));
    let c = a.mul(&b);
    let d = b.add(&c);
    d.backward();

    let b_grad = b.grad.borrow();
    println!("b_grad: {:?}", b_grad);

    let a_grad = a.grad.borrow();
    println!("a_grad: {:?}", a_grad);

    let c_grad = c.grad.borrow();
    println!("c_grad: {:?}", c_grad);

    let d_grad = d.grad.borrow();
    println!("d_grad: {:?}", d_grad);
}