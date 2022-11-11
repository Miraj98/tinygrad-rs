pub mod unary_ops;
pub mod binary_ops;
pub mod reduce_ops;

use ndarray::Array2;
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
    fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
           OpType::BinaryOp(a) => a.__backward(incoming_grad),
           OpType::UnaryOp(a) => a.__backward(incoming_grad),
           OpType::ReduceOp(a) => a.__backward(incoming_grad),
           OpType::Noop => {}
        };
    }
}

pub trait OpFunction {
    type Output;
    fn forward(&self, requires_grad: bool) -> Self::Output;
    fn backward(&self, incoming_grad: &Array2<f64>);
}