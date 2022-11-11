use ndarray::Array2;
use std::{
    cell::{Cell, UnsafeCell},
    rc::Rc,
};

use super::{OpFunction, OpType};
use crate::tensor::Tensor;

#[derive(Debug)]
pub enum BinaryOpType {
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Matmul(Matmul),
}

impl BinaryOpType {
    pub fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
            BinaryOpType::Add(a) => a.backward(incoming_grad),
            BinaryOpType::Sub(a) => a.backward(incoming_grad),
            BinaryOpType::Mul(a) => a.backward(incoming_grad),
            BinaryOpType::Matmul(a) => a.backward(incoming_grad),
        };
    }
}

pub trait BinaryOps {
    type Value;
    fn add(&self, x: &Self::Value) -> Rc<Tensor>;
    fn mul(&self, x: &Self::Value) -> Self::Value;
    fn sub(&self, x: &Self::Value) -> Self::Value;
    fn matmul(&self, x: &Self::Value) -> Self::Value;
}

#[derive(Debug)]
pub struct Add {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Add {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = &self.lhs.ndarray() as &Array2<f64> + &self.rhs.ndarray() as &Array2<f64>;
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::BinaryOp(BinaryOpType::Add(Add {
                    lhs: Rc::clone(&self.lhs),
                    rhs: Rc::clone(&self.rhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        if let Some(curr_grad_lhs) = self.lhs.grad().as_ref() {
            let grad = curr_grad_lhs + incoming_grad;
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(incoming_grad.to_owned()));
        }

        if let Some(curr_grad_rhs) = self.rhs.grad().as_ref() {
            let grad = curr_grad_rhs + incoming_grad;
            self.rhs.update_grad(Some(grad));
        } else {
            self.rhs.update_grad(Some(incoming_grad.to_owned()));
        }
    }
}
impl Add {
    pub fn from(a: &Rc<Tensor>, b: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
            rhs: Rc::clone(b),
        }
    }
}

#[derive(Debug)]
pub struct Sub {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Sub {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = &self.lhs.ndarray() as &Array2<f64> - &self.rhs.ndarray() as &Array2<f64>;
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::BinaryOp(BinaryOpType::Sub(Sub {
                    lhs: Rc::clone(&self.lhs),
                    rhs: Rc::clone(&self.rhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        if let Some(curr_grad_lhs) = self.lhs.grad().as_ref() {
            let grad = curr_grad_lhs + incoming_grad;
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some((incoming_grad).to_owned()));
        }

        if let Some(curr_grad_rhs) = self.rhs.grad().as_ref() {
            let grad = curr_grad_rhs - incoming_grad;
            self.rhs.update_grad(Some(grad));
        } else {
            self.rhs.update_grad(Some((-incoming_grad).to_owned()));
        }
    }
}
impl Sub {
    pub fn from(a: &Rc<Tensor>, b: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
            rhs: Rc::clone(b),
        }
    }
}

#[derive(Debug)]
pub struct Mul {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Mul {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = &self.lhs.ndarray() as &Array2<f64> * &self.rhs.ndarray() as &Array2<f64>;
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::BinaryOp(BinaryOpType::Mul(Mul {
                    lhs: Rc::clone(&self.lhs),
                    rhs: Rc::clone(&self.rhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let rhs = &self.rhs.ndarray() as &Array2<f64>;
        let lhs = &self.lhs.ndarray() as &Array2<f64>;

        if let Some(curr_grad_lhs) = self.lhs.grad().as_ref() {
            let grad = curr_grad_lhs + (rhs * incoming_grad);
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(rhs * incoming_grad));
        }

        if let Some(curr_grad_rhs) = self.rhs.grad().as_ref() {
            let grad = curr_grad_rhs - (lhs * incoming_grad);
            self.rhs.update_grad(Some(grad));
        } else {
            self.rhs.update_grad(Some(lhs * incoming_grad));
        }
    }
}
impl Mul {
    pub fn from(a: &Rc<Tensor>, b: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
            rhs: Rc::clone(b),
        }
    }
}

#[derive(Debug)]
pub struct Matmul {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Matmul {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        let a = (&self.lhs.ndarray() as &Array2<f64>).dot(&self.rhs.ndarray() as &Array2<f64>);
        Rc::new(Tensor {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::BinaryOp(BinaryOpType::Matmul(Matmul {
                    lhs: Rc::clone(&self.lhs),
                    rhs: Rc::clone(&self.rhs),
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        let rhs = self.rhs.ndarray();
        let lhs = self.lhs.ndarray();
        let rhs_t = rhs.t();
        let lhs_t = lhs.t();

        if let Some(curr_grad_lhs) = self.lhs.grad().as_ref() {
            let grad = curr_grad_lhs + (incoming_grad.dot(&rhs_t));
            self.lhs.update_grad(Some(grad));
        } else {
            self.lhs.update_grad(Some(incoming_grad.dot(&rhs_t)));
        }

        if let Some(curr_grad_rhs) = self.rhs.grad().as_ref() {
            let grad = curr_grad_rhs - (lhs_t.dot(incoming_grad));
            self.rhs.update_grad(Some(grad));
        } else {
            self.rhs.update_grad(Some(lhs_t.dot(incoming_grad)));

        }
    }
}
impl Matmul {
    pub fn from(a: &Rc<Tensor>, b: &Rc<Tensor>) -> Self {
        Self {
            lhs: Rc::clone(a),
            rhs: Rc::clone(b),
        }
    }
}
