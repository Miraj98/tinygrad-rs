use std::rc::Rc;
use crate::tensor::Tensor;
use super::OpFunction;

#[derive(Debug)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Matmul,
}

pub trait BinaryOps {
    type Value;
    fn add(&self, x: &Self::Value) -> Self::Value;
    fn mul(&self, x: &Self::Value) -> Self::Value;
    fn sub(&self, x: &Self::Value) -> Self::Value;
    fn matmul(&self, x: &Self::Value) -> Self::Value;
}

pub struct Add {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Add {
    type Output = Rc<Tensor>;
    
    fn forward(&self) -> Self::Output {
        self.lhs.add(&self.rhs)
    }

    fn backward(&self) {
       todo!() 
    }
}

pub struct Sub {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Sub {
    type Output = Rc<Tensor>;
    
    fn forward(&self) -> Self::Output {
        self.lhs.sub(&self.rhs)
    }

    fn backward(&self) {
       todo!() 
    }
}

pub struct Mul {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Mul {
    type Output = Rc<Tensor>;
    
    fn forward(&self) -> Self::Output {
        self.lhs.mul(&self.rhs)
    }

    fn backward(&self) {
       todo!() 
    }
}

pub struct Matmul {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
}
impl OpFunction for Matmul {
    type Output = Rc<Tensor>;
    
    fn forward(&self) -> Self::Output {
        self.lhs.matmul(&self.rhs)
    }

    fn backward(&self) {
       todo!() 
    }
}