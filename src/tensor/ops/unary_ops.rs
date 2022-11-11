use std::{rc::Rc, cell::{UnsafeCell, Cell}};

use crate::tensor::Tensor;

use super::{OpFunction, OpType};

#[derive(Debug)]
pub enum UnaryOpType {
    Square(Square),
    Sigmoid(Sigmoid)
}

pub trait UnaryOps {
    type Value;
    fn sigmoid(&self) -> Self::Value;
    fn square(&self) -> Self::Value;
}

#[derive(Debug)]
pub struct Sigmoid {
    lhs: Rc<Tensor>
}
impl OpFunction for Sigmoid {
    type Output = Rc<Tensor>;

     fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self.lhs.ndarray().mapv(|val| 1.0 / (1.0 + f64::exp(-val)));
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::UnaryOp(UnaryOpType::Sigmoid(Sigmoid {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
     }

    fn backward(&self) {
        todo!()
    }
}
impl Sigmoid {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
        }
    }
}

#[derive(Debug)]
pub struct Square {
    lhs: Rc<Tensor>
}
impl OpFunction for Square {
    type Output = Rc<Tensor>;

     fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self.lhs.ndarray().mapv(|val| val * val);
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::UnaryOp(UnaryOpType::Square(Square {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
     }

    fn backward(&self) {
        todo!()
    }
}
impl Square {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
        }
    }
}
