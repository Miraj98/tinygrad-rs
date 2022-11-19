pub mod unary_ops;
pub mod binary_ops;
pub mod reduce_ops;
pub mod processing_ops;

use ndarray::Array2;
use unary_ops::*;
use binary_ops::*;
use reduce_ops::*;
use processing_ops::*;

pub enum OperationType<T: OpFunction> {
    // Binary ops
    Add(T),
    Sub(T),
    Mul(T),
    Matmul(T),

    // Unary ops
    Square(T),
    Sigmoid(T),
    ReLU(T),
    NaturalLog(T),

    // Reduce ops
    Mean(T),
    Sum(T),

    // Processing ops
    Conv2d(T)
}

#[derive(Debug)]
pub enum OpType {
    BinaryOp(BinaryOpType),
    UnaryOp(UnaryOpType),
    ReduceOp(ReduceOpType),
    ProcessingOp(ProcessingOpType),
    Noop,
}

impl OpType {
    pub fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
           OpType::BinaryOp(a) => a.__backward(incoming_grad),
           OpType::UnaryOp(a) => a.__backward(incoming_grad),
           OpType::ReduceOp(a) => a.__backward(incoming_grad),
           OpType::ProcessingOp(a) => a.__backward(incoming_grad),
           OpType::Noop => {}
        };
    }
}

pub trait OpFunction {
    type Output;
    fn forward(&self, requires_grad: bool) -> Self::Output;
    fn backward(&self, incoming_grad: &Array2<f64>);
}