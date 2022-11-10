#[derive(Debug)]
pub enum UnaryOpType {
    Square,
    Sigmoid
}

pub trait UnaryOps {
    type Value;
    fn sigmoid(&self) -> Self::Value;
    fn square(&self) -> Self::Value;
}
