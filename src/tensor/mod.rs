pub mod ops;

use std::cell::{UnsafeCell, Cell};
use ndarray::Array2;
use ops::OpFunction;

enum RefState {
    Exclusive,
    Shared(u32),
    Unshared,
}

pub struct Tensor<Op: OpFunction> {
    pub data: UnsafeCell<Array2<f64>>,
    pub grad: UnsafeCell<Array2<f64>>,
    __grad_ref: Cell<RefState>,
    __data_ref: Cell<RefState>,
    ctx: Option<Op>
}

impl<Op: OpFunction> Tensor<Op> {
    pub fn grad(&self) -> &Array2<f64> {
        todo!("Increase __grad_ref reference count");
        unsafe {& *self.grad.get() }
    }

    pub fn dim(&self) -> (usize, usize) {
        todo!()
    }

    pub fn ndarray(&self) -> &Array2<f64> {
        todo!()
    }
}

impl<Op: OpFunction> Drop for Tensor<Op> {
    fn drop(&mut self) {
        todo!()
    }
}
