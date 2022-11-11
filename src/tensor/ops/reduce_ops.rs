use std::{rc::Rc, cell::{UnsafeCell, Cell}};

use ndarray::Array2;

use crate::tensor::Tensor;

use super::{OpFunction, OpType};

#[derive(Debug)]
pub enum ReduceOpType {
    Mean(Mean),
    Sum(Sum),
}

impl ReduceOpType {
    pub fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
            ReduceOpType::Sum(a) => a.backward(incoming_grad),
            ReduceOpType::Mean(a) => a.backward(incoming_grad)
        };
    }
}

pub trait ReduceOps {
    type Value;
    fn sum(&self) -> Self::Value;
    fn mean(&self) -> Self::Value;
}

#[derive(Debug)]
pub struct Mean {
    lhs: Rc<Tensor>
}
impl OpFunction for Mean {
    type Output = Rc<Tensor>;
    
    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self.lhs.ndarray().mean().unwrap();
        Rc::new(Tensor {
            data: UnsafeCell::new(&Array2::ones((1, 1)) * a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::ReduceOp(ReduceOpType::Mean(Mean {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        todo!()
    }
}
impl Mean {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
        }
    }
}

#[derive(Debug)]
pub struct Sum {
    lhs: Rc<Tensor>
}
impl OpFunction for Sum {
    type Output = Rc<Tensor>;
    
    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self.lhs.ndarray().sum();
        Rc::new(Tensor {
            data: UnsafeCell::new(&Array2::ones((1, 1)) * a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::ReduceOp(ReduceOpType::Sum(Sum {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        todo!()
    }
}
impl Sum {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
        }
    }
}