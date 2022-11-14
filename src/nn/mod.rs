use crate::tensor::{
    ops::{binary_ops::BinaryOps, unary_ops::UnaryOps, reduce_ops::ReduceOps},
    Tensor,
};
use std::rc::Rc;

pub fn cross_entropy(input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
    assert_eq!(input.dim(), target.dim());
    let lhs = target.mul(&input.ln());

    let ones = Tensor::ones(input.dim(), None);
    let rhs = ones.sub(&target).mul(&ones.sub(&input).ln());

    let l = lhs.add(&rhs);
    let loss = l.sum();
    loss
}

pub fn quadratic_loss(input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
    assert_eq!(input.dim(), target.dim());
    let l = target.sub(&input).square().mul_scalar(0.5);
    let loss = l.mean();
    loss
}
