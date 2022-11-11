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
    // Sub(Sub<Op>),
    // Mul(Mul<Op>),
    // Matmul(Matmul<Op>),
}

pub trait BinaryOps {
    type Value;
    fn add(&self, x: &Self::Value) -> Rc<Tensor>;
    // fn mul(&self, x: &Self::Value) -> Self::Value;
    // fn sub(&self, x: &Self::Value) -> Self::Value;
    // fn matmul(&self, x: &Self::Value) -> Self::Value;
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

    fn backward(&self) {
        todo!()
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

// pub struct Sub {
//     lhs: Rc<Tensor<Self>>,
//     rhs: Rc<Tensor<Self>>,
// }
// impl OpFunction for Sub {
//     fn backward(&self) {
//        todo!()
//     }
// }
// impl Sub {
//     pub fn new(a: Rc<Tensor<Self>>, b: Rc<Tensor<Self>>) -> Self {
//         Self  { lhs: a, rhs: b }
//     }

//     pub fn from(a: &Rc<Tensor<Self>>, b: &Rc<Tensor<Self>>) -> Self {
//         Self  { lhs: Rc::clone(a), rhs: Rc::clone(b) }
//     }
// }

// pub struct Mul {
//     lhs: Rc<Tensor<Self>>,
//     rhs: Rc<Tensor<Self>>,
// }
// impl OpFunction for Mul {
//     fn backward(&self) {
//        todo!()
//     }
// }
// impl Mul {
//     pub fn new(a: Rc<Tensor<Self>>, b: Rc<Tensor<Self>>) -> Self {
//         Self  { lhs: a, rhs: b }
//     }

//     pub fn from(a: &Rc<Tensor<Self>>, b: &Rc<Tensor<Self>>) -> Self {
//         Self  { lhs: Rc::clone(a), rhs: Rc::clone(b) }
//     }
// }

// pub struct Matmul {
//     lhs: Rc<Tensor<Self>>,
//     rhs: Rc<Tensor<Self>>,
// }
// impl OpFunction for Matmul {
//     fn backward(&self) {
//        todo!()
//     }
// }
// impl Matmul {
//     pub fn new(a: Rc<Tensor<Self>>, b: Rc<Tensor<Self>>) -> Self {
//         Self  { lhs: a, rhs: b }
//     }
// }
