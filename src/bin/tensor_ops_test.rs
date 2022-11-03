use tinygrad_rust::tensor::Tensor;
use ndarray::arr2;

fn main() {
    let mut a = Tensor::new(arr2(&[[3.]]));
    let mut b = Tensor::new(arr2(&[[12.]]));
    let mut c = a.mul(&mut b);
    c.backward();
}