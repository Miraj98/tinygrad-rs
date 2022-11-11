pub mod ops;
pub mod tensor_ref;

use ndarray::Array2;
use ops::binary_ops::Add;
use ops::OpFunction;
use std::collections::HashSet;
use std::rc::Rc;
use std::{
    cell::{Cell, UnsafeCell},
    ptr::NonNull,
};
use tensor_ref::{BorrowRef, Ref};
use self::ops::binary_ops::{BinaryOpType, BinaryOps, Matmul, Mul, Sub};
use self::ops::reduce_ops::{Mean, ReduceOpType, ReduceOps, Sum};
use self::ops::unary_ops::{Sigmoid, Square, UnaryOpType, UnaryOps};
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

    pub fn update_grad(&self, grad: Option<Array2<f64>>) {
        unsafe { *self.grad_value.get() = grad };
    }

    fn __deepwalk(&self) -> Vec<*const Tensor> {
        let mut visited = HashSet::<*const Tensor>::new();
        let mut added = HashSet::<*const Tensor>::new();
        let mut work_stack = Vec::<*const Tensor>::new();

        let mut topo = Vec::<*const Tensor>::new();
        work_stack.push(self as *const Tensor);

        while work_stack.len() > 0 {
            if let Some(t) = work_stack.pop() {
                if visited.contains(&t) {
                    if !added.contains(&t) {
                        topo.push(t);
                        added.insert(t);
                    }
                } else {
                    visited.insert(t);
                    unsafe {
                        match &(*t).ctx {
                            OpType::Noop => {
                                if !added.contains(&t) {
                                    topo.push(t);
                                    added.insert(t);
                                }
                            }
                            OpType::BinaryOp(BinaryOpType::Add(a)) => {
                                work_stack.push(t);
                                let (lhs_ptr, rhs_ptr) = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }

                                if !visited.contains(&rhs_ptr) {
                                    work_stack.push(rhs_ptr);
                                }
                            }
                            OpType::BinaryOp(BinaryOpType::Sub(a)) => {
                                work_stack.push(t);
                                let (lhs_ptr, rhs_ptr) = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }

                                if !visited.contains(&rhs_ptr) {
                                    work_stack.push(rhs_ptr);
                                }
                            }
                            OpType::BinaryOp(BinaryOpType::Mul(a)) => {
                                work_stack.push(t);
                                let (lhs_ptr, rhs_ptr) = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }

                                if !visited.contains(&rhs_ptr) {
                                    work_stack.push(rhs_ptr);
                                }
                            }
                            OpType::BinaryOp(BinaryOpType::Matmul(a)) => {
                                work_stack.push(t);
                                let (lhs_ptr, rhs_ptr) = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }

                                if !visited.contains(&rhs_ptr) {
                                    work_stack.push(rhs_ptr);
                                }
                            }
                            OpType::UnaryOp(UnaryOpType::Sigmoid(a)) => {
                                work_stack.push(t);
                                let lhs_ptr = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }
                            }
                            OpType::UnaryOp(UnaryOpType::Square(a)) => {
                                work_stack.push(t);
                                let lhs_ptr = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }
                            }
                            OpType::ReduceOp(ReduceOpType::Mean(a)) => {
                                work_stack.push(t);
                                let lhs_ptr = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }
                            }
                            OpType::ReduceOp(ReduceOpType::Sum(a)) => {
                                work_stack.push(t);
                                let lhs_ptr = a.get_raw_ptr();
                                if !visited.contains(&lhs_ptr) {
                                    work_stack.push(lhs_ptr);
                                }
                            }
                        }
                    }
                }
            }
        }

        topo.reverse();
        topo
    }

    fn __backward(&self, incoming_grad: &Array2<f64>) {
        self.ctx.__backward(incoming_grad);
    }

    pub fn backward(&self) {
        let tensors = self.__deepwalk();
        let self_grad = Array2::<f64>::ones(self.dim());

        for t in &tensors {
            unsafe {
                let pg = (**t).grad();
                if let Some(p_grad) = pg.as_ref() {
                    (**t).__backward(p_grad);
                } else {
                    (**t).__backward(&self_grad);
                }
            }
            
        }

        self.update_grad(Some(self_grad));
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

impl ReduceOps for Rc<Tensor> {
    type Value = Rc<Tensor>;

    fn mean(&self) -> Self::Value {
        let requires_grad = self.requires_grad.get().unwrap_or(false);
        let op = Mean::from(self);
        let output = op.forward(requires_grad);
        output
    }

    fn sum(&self) -> Self::Value {
        let requires_grad = self.requires_grad.get().unwrap_or(false);
        let op = Sum::from(self);
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
    fn add_tensors_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let b = Tensor::new(array![[2., 3.], [4., 5.]], Some(true));
        let out = a.add(&b);

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
        assert_eq!(b.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
    }

    #[test]
    fn sub_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = b.sub(&a);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 1.], [1., 1.]]);
    }

    #[test]
    fn sub_tensors_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let b = Tensor::new(array![[2., 3.], [4., 5.]], Some(true));
        let out = b.sub(&a);

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[-1., -1.], [-1., -1.]]);
        assert_eq!(b.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
    }

    #[test]
    fn mul_tensors() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
        let out = a.mul(&b);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[2., 6.], [12., 20.]]);
    }

    #[test]
    fn mul_tensors_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let b = Tensor::new(array![[2., 3.], [4., 5.]], Some(true));
        let out = a.mul(&b);

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[2., 3.], [4., 5.]]);
        assert_eq!(b.grad().as_ref().unwrap(), array![[1., 2.], [3., 4.]]);
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

    #[test]
    fn matmul_tensors_grad_test() {
        let a = Tensor::new(array![[1., 2., 3.], [4., 5., 6.]], Some(true));
        let b = Tensor::new(array![[2., 3.], [4., 5.], [6., 7.]], Some(true));
        let out = a.matmul(&b);

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(
            a.grad().as_ref().unwrap(),
            array![[5., 9., 13.], [5., 9., 13.]]
        );
        assert_eq!(
            b.grad().as_ref().unwrap(),
            array![[5., 5.], [7., 7.], [9., 9.]]
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
    fn sigmoid_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let out = a.sigmoid();

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(
            a.grad().as_ref().unwrap(),
            array![
                [0.19661193324148188, 0.1049935854035065],
                [0.045176659730912144, 0.017662706213291114]
            ]
        );
    }

    #[test]
    fn square_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let out = a.square();
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 4.], [9., 16.]]);
    }

    #[test]
    fn square_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let out = a.square();

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out.__backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[2., 4.], [6., 8.]]);
    }
}

#[cfg(test)]
mod reduce_ops_tests {
    use crate::tensor::{ops::reduce_ops::ReduceOps, Tensor};
    use ndarray::{array, Array2};

    #[test]
    fn mean_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let out = a.mean();
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[2.5]]);
    }

    #[test]
    fn mean_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let out = a.mean();

        let out_grad_array = array![[1.]];
        out.__backward(&out_grad_array);

        assert_eq!(
            a.grad().as_ref().unwrap(),
            array![[0.25, 0.25], [0.25, 0.25]]
        );
    }

    #[test]
    fn sum_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
        let out = a.sum();
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[10.]]);
    }

    #[test]
    fn sum_grad_test() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let out = a.sum();

        let out_grad_array = array![[1.]];
        out.__backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
    }
}

#[cfg(test)]
mod autodiff_tests {
    use ndarray::array;

    use super::{Tensor, ops::binary_ops::BinaryOps};

    #[test]
    fn topo_order() {
        let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
        let b = Tensor::new(array![[5., 6.], [7., 8.]], Some(true));
        let c = a.add(&b);
        let d = c.mul(&a);
        
        let order = d.__deepwalk();

        assert_eq!(order.len(), 4);
        assert_eq!(order[0], d.as_ref() as *const Tensor);
        assert_eq!(order[1], c.as_ref() as *const Tensor);
        assert_eq!(order[2], b.as_ref() as *const Tensor);
        assert_eq!(order[3], a.as_ref() as *const Tensor);
    }
}
