use crate::tensor::{TensorBase, TensorBaseImpl};
use ndarray::{Array, Dimension};
use std::rc::Rc;

pub trait BinaryOps {
    fn add(&self, a: &Self) -> Self;
    // fn sub(&self, a: Self) -> Self;
    // fn mul(&self, a: Self) -> Self;
    // fn matmul(&self, a: Self) -> Self;
}

impl<D> BinaryOps for Rc<TensorBase<f64, D>>
where
    D: Dimension + 'static,
{
    fn add(&self, a: &Self) -> Self {
        let lhs = &self.ndarray() as &Array<f64, D>;
        let rhs = &a.ndarray() as &Array<f64, D>;
        let out = lhs + rhs;

        let lhs_clone = Rc::clone(self);
        let rhs_clone = Rc::clone(a);
        let ret = TensorBase::construct_with_backward_fn(
            out,
            move |incoming_grad: &Array<f64, D>| {
                if let Some(curr_grad_lhs) = lhs_clone.grad().as_ref() {
                    let grad = curr_grad_lhs + incoming_grad;
                    lhs_clone.update_grad(Some(grad));
                } else {
                    lhs_clone.update_grad(Some((incoming_grad).to_owned()));
                }

                if let Some(curr_grad_rhs) = rhs_clone.grad().as_ref() {
                    let grad = curr_grad_rhs + incoming_grad;
                    rhs_clone.update_grad(Some(grad));
                } else {
                    rhs_clone.update_grad(Some((incoming_grad).to_owned()));
                }
            },
            None,
        );
        ret
    }
}
