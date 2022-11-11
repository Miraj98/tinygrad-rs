pub mod ops;
pub mod tensor_ref;

use ndarray::Array2;
use ops::binary_ops::Add;
use ops::OpFunction;
use std::rc::Rc;
use std::{
    cell::{Cell, UnsafeCell},
    ptr::NonNull,
};
use tensor_ref::{BorrowRef, Ref};

use self::ops::binary_ops::{BinaryOps, Matmul, Mul, Sub};
use self::ops::unary_ops::{Sigmoid, Square, UnaryOps};
use self::ops::OpType;

#[derive(Debug)]
pub struct Tensor {
    data: UnsafeCell<Array2<f64>>,
    grad_value: UnsafeCell<Option<Array2<f64>>>,
    grad_borrow: Cell<isize>,
    data_borrow: Cell<isize>,
    ctx: OpType,
    requires_grad: Cell<Option<bool>>,
}

impl Tensor {
    fn __new(a: Array2<f64>, op: OpType, requires_grad: Option<bool>) -> Rc<Tensor> {
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: op,
            requires_grad: Cell::new(requires_grad),
        })
    }

    pub fn new(a: Array2<f64>, requires_grad: Option<bool>) -> Rc<Tensor> {
        Tensor::__new(a, OpType::Noop, requires_grad)
    }

    pub fn try_grad(&self) -> Result<Ref<Option<Array2<f64>>>, isize> {
        match BorrowRef::new(&self.grad_borrow) {
            Some(b) => {
                let value = unsafe { NonNull::new_unchecked(self.grad_value.get()) };
                Ok(Ref { value, borrow: b })
            }
            None => Err(self.grad_borrow.get()),
        }
    }

    pub fn grad(&self) -> Ref<'_, Option<Array2<f64>>> {
        self.try_grad().expect("already mutably borrowed")
    }

    pub fn dim(&self) -> (usize, usize) {
        unsafe { (*self.data.get()).dim() }
    }

    pub fn try_ndarray(&self) -> Result<Ref<Array2<f64>>, isize> {
        match BorrowRef::new(&self.data_borrow) {
            Some(b) => {
                let value = unsafe { NonNull::new_unchecked(self.data.get()) };
                Ok(Ref { value, borrow: b })
            }
            None => Err(self.data_borrow.get()),
        }
    }

    pub fn ndarray(&self) -> Ref<'_, Array2<f64>> {
        self.try_ndarray().expect("already mutably borrowed")
    }
}

impl BinaryOps for Rc<Tensor> {
    type Value = Rc<Tensor>;

    fn add(&self, x: &Self::Value) -> Self::Value {
        let requires_grad =
            self.requires_grad.get().unwrap_or(false) || x.requires_grad.get().unwrap_or(false);
        let op = Add::from(self, x);
        let output = op.forward(requires_grad);
        output
    }

    fn sub(&self, x: &Self::Value) -> Self::Value {
        let requires_grad =
            self.requires_grad.get().unwrap_or(false) || x.requires_grad.get().unwrap_or(false);
        let op = Sub::from(self, x);
        let output = op.forward(requires_grad);
        output
    }

    fn mul(&self, x: &Self::Value) -> Self::Value {
        let requires_grad =
            self.requires_grad.get().unwrap_or(false) || x.requires_grad.get().unwrap_or(false);
        let op = Mul::from(self, x);
        let output = op.forward(requires_grad);
        output
    }

    fn matmul(&self, x: &Self::Value) -> Self::Value {
        let requires_grad =
            self.requires_grad.get().unwrap_or(false) || x.requires_grad.get().unwrap_or(false);
        let op = Matmul::from(self, x);
        let output = op.forward(requires_grad);
        output
    }
}

impl UnaryOps for Rc<Tensor> {
    type Value = Rc<Tensor>;
    fn sigmoid(&self) -> Self::Value {
        let requires_grad = self.requires_grad.get().unwrap_or(false);
        let op = Sigmoid::from(self);
        let output = op.forward(requires_grad);
        output
    }

    fn square(&self) -> Self::Value {
        let requires_grad = self.requires_grad.get().unwrap_or(false);
        let op = Square::from(self);
        let output = op.forward(requires_grad);
        output
    }
}

#[cfg(test)]
mod binary_ops_tests {
    use super::{ops::binary_ops::BinaryOps, Tensor};
    use ndarray::{array, Array2};

    #[test]
    fn add_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = a.add(&b);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[3., 5.], [7., 9.]]);
    }

    #[test]
    fn sub_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = b.sub(&a);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 1.], [1., 1.]]);
    }

    #[test]
    fn mul_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = a.mul(&b);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[2., 6.], [12., 20.]]);
    }

    #[test]
    fn matmul_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = a.matmul(&b);
        assert_eq!(
            &out.ndarray() as &Array2<f64>,
            array![[10., 13.], [22., 29.]]
        );
    }
}

#[cfg(test)]
mod unary_ops_tests {
    use crate::tensor::{ops::unary_ops::UnaryOps, Tensor};
    use ndarray::{array, Array2};

    #[test]
    fn sigmoid_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let out = a.sigmoid();
        assert_eq!(
            &out.ndarray() as &Array2<f64>,
            array![
                [0.7310585786300049, 0.8807970779778823],
                [0.9525741268224334, 0.9820137900379085]
            ]
        );
    }

    #[test]
    fn square_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let out = a.square();
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 4.], [9., 16.]]);
    }
}
