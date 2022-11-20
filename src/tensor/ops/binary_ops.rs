use crate::tensor::{TensorBase, TensorBaseImpl};
use ndarray::{Array, Dim, Dimension};
use std::rc::Rc;

pub trait BinaryOps {
    fn add(&self, a: &Self) -> Self;
    fn sub(&self, a: &Self) -> Self;
    fn mul(&self, a: &Self) -> Self;
}

pub trait MatMul<A>
where
    A: Clone,
{
    fn matmul(&self, a: &Rc<TensorBase<A, Dim<[usize; 2]>>>) -> Self;
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
                let lhs_grad: Array<f64, D>;
                if let Some(curr_grad_lhs) = lhs_clone.grad().as_ref() {
                    lhs_grad = curr_grad_lhs + incoming_grad;
                } else {
                    lhs_grad = incoming_grad.to_owned();
                }
                lhs_clone._backward(&lhs_grad);
                lhs_clone.update_grad(Some(lhs_grad));

                let rhs_grad: Array<f64, D>;
                if let Some(curr_grad_rhs) = rhs_clone.grad().as_ref() {
                    rhs_grad = curr_grad_rhs + incoming_grad;
                } else {
                    rhs_grad = incoming_grad.to_owned();
                }
                rhs_clone._backward(&rhs_grad);
                rhs_clone.update_grad(Some(rhs_grad));
            },
            None,
        );
        ret
    }

    fn sub(&self, a: &Self) -> Self {
        let lhs = &self.ndarray() as &Array<f64, D>;
        let rhs = &a.ndarray() as &Array<f64, D>;
        let out = lhs - rhs;

        let lhs_clone = Rc::clone(self);
        let rhs_clone = Rc::clone(a);
        let ret = TensorBase::construct_with_backward_fn(
            out,
            move |incoming_grad: &Array<f64, D>| {
                let lhs_grad: Array<f64, D>;
                if let Some(curr_grad_lhs) = lhs_clone.grad().as_ref() {
                    lhs_grad = curr_grad_lhs + incoming_grad;
                } else {
                    lhs_grad = incoming_grad.to_owned();
                }
                lhs_clone._backward(&lhs_grad);
                lhs_clone.update_grad(Some(lhs_grad));

                let rhs_grad: Array<f64, D>;
                if let Some(curr_grad_rhs) = rhs_clone.grad().as_ref() {
                    rhs_grad = curr_grad_rhs - incoming_grad;
                } else {
                    rhs_grad = (-incoming_grad).to_owned();
                }
                rhs_clone._backward(&rhs_grad);
                rhs_clone.update_grad(Some(rhs_grad));
            },
            None,
        );
        ret
    }

    fn mul(&self, a: &Self) -> Self {
        let lhs = &self.ndarray() as &Array<f64, D>;
        let rhs = &a.ndarray() as &Array<f64, D>;
        let out = lhs * rhs;

        let lhs_clone = Rc::clone(self);
        let rhs_clone = Rc::clone(a);
        let ret = TensorBase::construct_with_backward_fn(
            out,
            move |incoming_grad: &Array<f64, D>| {
                let lhs = &lhs_clone.ndarray() as &Array<f64, D>;
                let rhs = &rhs_clone.ndarray() as &Array<f64, D>;

                let lhs_grad: Array<f64, D>;
                if let Some(curr_grad_lhs) = lhs_clone.grad().as_ref() {
                    lhs_grad = curr_grad_lhs + incoming_grad * rhs;
                } else {
                    lhs_grad = incoming_grad * rhs;
                }
                lhs_clone._backward(&lhs_grad);
                lhs_clone.update_grad(Some(lhs_grad));

                let rhs_grad: Array<f64, D>;
                if let Some(curr_grad_rhs) = rhs_clone.grad().as_ref() {
                    rhs_grad = curr_grad_rhs + incoming_grad * lhs;
                } else {
                    rhs_grad = incoming_grad * lhs;
                }
                rhs_clone._backward(&rhs_grad);
                rhs_clone.update_grad(Some(rhs_grad));
            },
            None,
        );
        ret
    }
}

impl MatMul<f64> for Rc<TensorBase<f64, Dim<[usize; 2]>>> {
    fn matmul(&self, a: &Rc<TensorBase<f64, Dim<[usize; 2]>>>) -> Self {
        let lhs = &self.ndarray() as &Array<f64, Dim<[usize; 2]>>;
        let rhs = &a.ndarray() as &Array<f64, Dim<[usize; 2]>>;
        let out = lhs.dot(rhs);

        let lhs_clone = Rc::clone(self);
        let rhs_clone = Rc::clone(a);
        let ret = TensorBase::construct_with_backward_fn(
            out,
            move |incoming_grad: &Array<f64, Dim<[usize; 2]>>| {
                let lhs = &lhs_clone.ndarray() as &Array<f64, Dim<[usize; 2]>>;
                let rhs = &rhs_clone.ndarray() as &Array<f64, Dim<[usize; 2]>>;

                let lhs_grad: Array<f64, Dim<[usize; 2]>>;
                if let Some(curr_grad_lhs) = lhs_clone.grad().as_ref() {
                    lhs_grad = curr_grad_lhs + incoming_grad.dot(&rhs.t());
                } else {
                    lhs_grad = incoming_grad.dot(&rhs.t());
                }
                lhs_clone._backward(&lhs_grad);
                lhs_clone.update_grad(Some(lhs_grad));

                let rhs_grad: Array<f64, Dim<[usize; 2]>>;
                if let Some(curr_grad_rhs) = rhs_clone.grad().as_ref() {
                    rhs_grad = curr_grad_rhs + lhs.t().dot(incoming_grad);
                } else {
                    rhs_grad = lhs.t().dot(incoming_grad);
                }
                rhs_clone._backward(&rhs_grad);
                rhs_clone.update_grad(Some(rhs_grad));
            },
            None,
        );
        ret
    }
}
