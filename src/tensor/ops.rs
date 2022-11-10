use crate::tensor::Tensor;
use ndarray::{Array2, ShapeBuilder, Dim};

#[derive(Debug)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Matmul,
}

#[derive(Debug)]
pub enum UnaryOpType {
    Square,
    Sigmoid
}

#[derive(Debug)]
pub enum ReduceOpTypes {
    Mean,
    Sum,
}

#[derive(Debug)]
pub enum OpType {
    BinaryOp(BinaryOpType),
    UnaryOp(UnaryOpType),
    ReduceOp(ReduceOpTypes)
}

pub trait TensorConstructors {
    fn new(a: Array2<f64>, requires_grad: bool) -> Tensor;
    fn ones<Sh>(shape: Sh, requires_grad: bool) -> Tensor where Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>;
    fn zeros<Sh>(shape: Sh, requires_grad: bool) -> Tensor where Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>;
    fn fill<Sh>(shape: Sh, x: f64, requires_grad: bool) -> Tensor where Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>;
    fn randn<Sh>(shape: Sh, requires_grad: bool) -> Tensor where Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>;
}

pub trait BinaryOps {
    fn add(&self, x: &Tensor) -> Tensor;
    fn mul(&self, x: &Tensor) -> Tensor;
    fn sub(&self, x: &Tensor) -> Tensor;
    fn matmul(&self, x: &Tensor) -> Tensor;
}

pub trait UnaryOps {
    fn sigmoid(&self) -> Tensor;
    fn square(&self) -> Tensor;
}

pub trait ReduceOps {
    fn sum(&self) -> Tensor;
    fn mean(&self) -> Tensor;
}

pub trait Backprop {
    fn backward(&self);
    fn zero_grad(&self);
}
