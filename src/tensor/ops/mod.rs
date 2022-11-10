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
    ReduceOp(ReduceOpType)
}

pub trait OpFunction {
    type Output;
    fn forward(&self) -> Self::Output;
    fn backward(&self);
}