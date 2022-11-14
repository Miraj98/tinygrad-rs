use std::{
    cell::{Cell, UnsafeCell},
    rc::Rc,
};

use ndarray::Array2;

use crate::tensor::Tensor;

use super::{OpFunction, OpType};

#[derive(Debug)]
pub enum UnaryOpType {
    Square(Square),
    Sigmoid(Sigmoid),
    ReLU(ReLU),
    NaturalLog(NaturalLog),
}

impl UnaryOpType {
    pub fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
            Self::Square(s) => s.backward(incoming_grad),
            Self::Sigmoid(s) => s.backward(incoming_grad),
            Self::ReLU(s) => s.backward(incoming_grad),
            Self::NaturalLog(s) => s.backward(incoming_grad),
        }
    }
}

pub trait UnaryOps {
    type Value;
    fn sigmoid(&self) -> Self::Value;
    fn square(&self) -> Self::Value;
    fn relu(&self) -> Self::Value;
    fn ln(&self) -> Self::Value;
}

#[derive(Debug)]
pub struct Sigmoid {
    lhs: Rc<Tensor>,
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

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let lhs = &self.lhs.ndarray() as &Array2<f64>;
        let local_grad =
            lhs.mapv(|val| f64::exp(-val) / ((1. + f64::exp(-val)) * (1. + f64::exp(-val))));
        if let Some(curr_grad) = self.lhs.grad().as_ref() {
            let grad = curr_grad + (local_grad * incoming_grad);
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(local_grad * incoming_grad));
        }
    }
}
impl Sigmoid {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self { lhs: Rc::clone(a) }
    }

    pub fn get_raw_ptr(&self) -> *const Tensor {
        self.lhs.as_ref() as *const Tensor
    }
}

#[derive(Debug)]
pub struct Square {
    lhs: Rc<Tensor>,
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

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let lhs = &self.lhs.ndarray() as &Array2<f64>;
        if let Some(curr_grad) = self.lhs.grad().as_ref() {
            let grad = curr_grad + (2. * lhs * incoming_grad);
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(2. * lhs * incoming_grad));
        }
    }
}
impl Square {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self { lhs: Rc::clone(a) }
    }

    pub fn get_raw_ptr(&self) -> *const Tensor {
        self.lhs.as_ref() as *const Tensor
    }
}

#[derive(Debug)]
pub struct ReLU {
    lhs: Rc<Tensor>,
}
impl OpFunction for ReLU {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self
            .lhs
            .ndarray()
            .mapv(|val| if val > 0. { val } else { 0. });
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::UnaryOp(UnaryOpType::ReLU(ReLU {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let lhs = &self.lhs.ndarray() as &Array2<f64>;
        let local_grad = lhs.mapv(|val| if val > 0. { 1. } else { 0. });
        if let Some(curr_grad) = self.lhs.grad().as_ref() {
            let grad = curr_grad + (local_grad * incoming_grad);
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(local_grad * incoming_grad));
        }
    }
}
impl ReLU {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self { lhs: Rc::clone(a) }
    }

    pub fn get_raw_ptr(&self) -> *const Tensor {
        self.lhs.as_ref() as *const Tensor
    }
}

#[derive(Debug)]
pub struct NaturalLog {
    lhs: Rc<Tensor>,
}
impl OpFunction for NaturalLog {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = self.lhs.ndarray().mapv(|val| val.ln());
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::UnaryOp(UnaryOpType::NaturalLog(NaturalLog {
                    lhs: Rc::clone(&self.lhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let lhs = &self.lhs.ndarray() as &Array2<f64>;
        let local_grad =
            lhs.mapv(|val| 1. / val);
        if let Some(curr_grad) = self.lhs.grad().as_ref() {
            let grad = curr_grad + (local_grad * incoming_grad);
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(local_grad * incoming_grad));
        }
    }
}
impl NaturalLog {
    pub fn from(a: &Rc<Tensor>) -> Self {
        Self { lhs: Rc::clone(a) }
    }

    pub fn get_raw_ptr(&self) -> *const Tensor {
        self.lhs.as_ref() as *const Tensor

    }
}
  
