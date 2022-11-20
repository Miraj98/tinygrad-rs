use super::TensorBase;
use ndarray::{DimMax, Dimension};
use std::rc::Rc;

// pub trait TensorOp<D> where D: Dimension {
//     type A: Clone;

//     fn forward(&self) -> Rc<TensorBase<Self::A, D>>;
//     fn backward(&self);
//     fn get_lhs<T>(&self) -> &Rc<TensorBase<Self::A, T>> where T: Dimension;
//     fn get_rhs<T>(&self) -> Option<&Rc<TensorBase<Self::A, T>>> where T: Dimension;
// }

// pub mod binary_ops;

// pub mod unary_ops;
pub mod binary_ops;
// pub mod reduce_ops;
// pub mod processing_ops;

// use ndarray::Array2;
// use unary_ops::*;
use binary_ops::*;
// use reduce_ops::*;
// use processing_ops::*;

#[derive(Debug)]
pub enum OpType<A, D1, D2>
where
    A: Clone,
    D1: Dimension + DimMax<D2>,
    D2: Dimension,
{
    BinaryOp(BinaryOpType<A, D1, D2>),
    // UnaryOp(UnaryOpType),
    // ReduceOp(ReduceOpType),
    // ProcessingOp(ProcessingOpType),
    Noop,
}

// impl OpType {
//     pub fn __backward(&self, incoming_grad: &Array2<f64>) {
//         match self {
//            OpType::BinaryOp(a) => a.__backward(incoming_grad),
//            OpType::UnaryOp(a) => a.__backward(incoming_grad),
//            OpType::ReduceOp(a) => a.__backward(incoming_grad),
//            OpType::ProcessingOp(a) => a.__backward(incoming_grad),
//            OpType::Noop => {}
//         };
//     }
// }

pub trait OpFunction {
    type Output;

    fn forward(&self, requires_grad: bool) -> Self::Output;
    fn backward(&self, incoming_grad: &Self::Output);
}
