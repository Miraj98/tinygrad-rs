use std::{
    cell::{Cell, UnsafeCell},
    rc::Rc,
};

use ndarray::{s, Array2, Axis};

use crate::tensor::Tensor;

use super::{OpFunction, OpType};

#[derive(Debug)]
pub enum ProcessingOpType {
    Conv2d(Conv2d),
}

impl ProcessingOpType {
    pub fn __backward(&self, incoming_grad: &Array2<f64>) {
        match self {
            ProcessingOpType::Conv2d(a) => a.backward(incoming_grad),
        };
    }
}

pub trait ProcessingOps {
    type Value;
    fn conv2d(&self, x: &Self::Value, strides: (usize, usize)) -> Rc<Tensor>;
}

#[derive(Debug)]
pub struct Conv2d {
    lhs: Rc<Tensor>,
    rhs: Rc<Tensor>,
    strides: (usize, usize),
}

impl OpFunction for Conv2d {
    type Output = Rc<Tensor>;

    fn forward(&self, requires_grad: bool) -> Self::Output {
        if self.rhs.ndarray().len() > self.lhs.ndarray().len() {
            panic!("Rhs size should be greater than Lhs in conv2d op")
        }
        let ox = (self.lhs.dim().0 - self.rhs.dim().0) / self.strides.0 + 1;
        let oy = (self.lhs.dim().1 - self.rhs.dim().1) / self.strides.1 + 1;

        let mut out_ndarray = Array2::<f64>::zeros((ox, oy));

        for x in 0..ox {
            for y in 0..oy {
                let o = &self.lhs.ndarray().slice(s![
                    x * self.strides.0..(x * self.strides.0 + self.rhs.dim().0),
                    y * self.strides.1..(y * self.strides.1 + self.rhs.dim().1),
                ]) * &self.rhs.ndarray() as &Array2<f64>;
                out_ndarray[(x, y)] += o.sum();
            }
        }

        Rc::new(Tensor {
            data: UnsafeCell::new(out_ndarray),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            ctx: if requires_grad {
                OpType::ProcessingOp(ProcessingOpType::Conv2d(Conv2d {
                    lhs: Rc::clone(&self.lhs),
                    rhs: Rc::clone(&self.rhs),
                    strides: self.strides,
                }))
            } else {
                OpType::Noop
            },
            requires_grad: Cell::new(Some(requires_grad)),
        })
    }

    fn backward(&self, incoming_grad: &Array2<f64>) {
        // padd lhs
        let px = self.rhs.ndarray().shape()[0] - 1;
        let py = self.rhs.ndarray().shape()[1] - 1;
        let mut padded_lhs = Array2::<f64>::zeros((
            2 * px + self.lhs.ndarray().dim().0,
            2 * py + self.lhs.ndarray().dim().1,
        ));
        padded_lhs
            .view_mut()
            .slice_mut(s![py..py + incoming_grad.dim().0, px..px + incoming_grad.dim().1])
            .assign(incoming_grad);
        
        // Rotate rhs view by 180deg
        let mut rhs_data = self.rhs.ndarray_mut();
        let mut rhs_view = rhs_data.view_mut();
        for i in 0..rhs_view.shape().len() {
            rhs_view.invert_axis(Axis(i));
        }

        let llhsx = self.lhs.dim().0;
        let llhsy = self.lhs.dim().1;
        let mut local_lhs_grad = Array2::<f64>::zeros((llhsx, llhsy));
        for x in 0..llhsx {
            for y in 0..llhsy {
                let o = &padded_lhs.slice(s![
                    x * self.strides.0..(x * self.strides.0 + rhs_view.dim().0),
                    y * self.strides.1..(y * self.strides.1 + rhs_view.dim().1),
                ]) * &rhs_view;
                local_lhs_grad[(x, y)] += o.sum();
            }
        }
        if let Some(curr_grad_lhs) = self.lhs.grad().as_ref() {
            self.lhs.update_grad(Some(curr_grad_lhs + local_lhs_grad));
        } else {
            self.lhs.update_grad(Some(local_lhs_grad));
        }

        let lrhsx = self.rhs.dim().0;
        let lrhsy = self.rhs.dim().1;
        let mut local_rhs_grad = Array2::<f64>::zeros((lrhsx, lrhsy));
        for x in 0..lrhsx {
            for y in 0..lrhsy {
                let o = &self.lhs.ndarray().slice(s![
                    x * self.strides.0..(x * self.strides.0 + incoming_grad.dim().0),
                    y * self.strides.1..(y * self.strides.1 + incoming_grad.dim().1),
                ]) * incoming_grad;
                local_rhs_grad[(x, y)] += o.sum();
            }
        }
        if let Some(curr_grad_rhs) = self.rhs.grad().as_ref() {
            self.rhs.update_grad(Some(curr_grad_rhs + local_rhs_grad));
        } else {
            self.rhs.update_grad(Some(local_rhs_grad));
        }


    }
}
impl Conv2d {
    pub fn from(a: &Rc<Tensor>, b: &Rc<Tensor>, strides: (usize, usize)) -> Self {
        Self {
            lhs: Rc::clone(a),
            rhs: Rc::clone(b),
            strides,
        }
    }

    pub fn get_raw_ptr(&self) -> (*const Tensor, *const Tensor) {
        (
            self.lhs.as_ref() as *const Tensor,
            self.rhs.as_ref() as *const Tensor,
        )
    }
}
