pub mod unary_ops;
pub mod binary_ops;
pub mod reduce_ops;

use unary_ops::*;
use binary_ops::*;
use reduce_ops::*;

#[derive(Debug)]
pub enum OpType {
    BinaryOp(BinaryOpType),
    UnaryOp(UnaryOpType),
    ReduceOp(ReduceOpType),
    Noop,
}

impl OpType {
    fn __backward(&self) {
        match self {
           OpType::BinaryOp(b) => b.__backward(),
           _ => {}
        };
    }
}

pub trait OpFunction {
    type Output;
    fn forward(&self, requires_grad: bool) -> Self::Output;
    fn backward(&self);
}