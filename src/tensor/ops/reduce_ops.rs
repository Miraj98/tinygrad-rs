#[derive(Debug)]
pub enum ReduceOpType {
    Mean,
    Sum,
}

pub trait ReduceOps {
    type Value;
    fn sum(&self) -> Self::Value;
    fn mean(&self) -> Self::Value;
}