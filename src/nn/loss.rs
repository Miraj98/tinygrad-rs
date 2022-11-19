use crate::tensor::{
    ops::{binary_ops::BinaryOps, reduce_ops::ReduceOps, unary_ops::UnaryOps},
    Tensor,
};
use std::rc::Rc;

pub trait Loss {
    fn calculate(&self, input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor>;
}

pub struct CrossEntropyLoss(pub &'static str);

impl Loss for CrossEntropyLoss {
    fn calculate(&self, input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
        assert_eq!(input.dim(), target.dim());
        let input_ln = input.ln();
        let lhs = target.mul(&input_ln);

        let ones = Tensor::ones(input.dim(), Some(true));
        let _a = ones.sub(&input);
        let _y = ones.sub(&target);
        let rhs = _y.mul(&_a.ln());

        let l = lhs.add(&rhs).mul_scalar(-1.);
        let loss = if self.0 == "mean" { l.mean() } else { l.sum() };
        loss
    }
}
pub struct MSELoss(pub &'static str);

impl Loss for MSELoss {
    fn calculate(&self, input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
        assert_eq!(input.dim(), target.dim());
        let l = target.sub(&input).square().mul_scalar(0.5);
        let loss = l.mean();
        loss
    }
}
