use ndarray::{Array2};
use std::{ops};

enum Op {
    Add,
    Sub,
    Mul
}

pub struct Tensor<'a> {
    data: Array2<f64>,
    prev: Option<Vec<&'a Tensor<'a>>>,
    next: Option<&'a Tensor<'a>>,
    op: Op,
    grad: Option<f64>,
}

impl<'a> ops::Add for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn add(self, rhs: Self) -> Tensor<'a> {
        let ret = Tensor {
            data: &self.data + &rhs.data,
            prev: Some(vec![self, rhs]),
            next: None,
            grad: None,
            op: Op::Add,
        };

        ret
    }
}

impl<'a> ops::Sub for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn sub(self, rhs: Self) -> Tensor<'a> {
        let ret = Tensor {
            data: &self.data - &rhs.data,
            prev: Some(vec![self, rhs]),
            next: None,
            grad: None,
            op: Op::Add,
        };

        ret
    }
}

impl<'a> ops::Mul for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn mul(self, rhs: Self) -> Tensor<'a> {
        let ret = Tensor {
            data: &self.data * &rhs.data,
            prev: Some(vec![self, rhs]),
            grad: None,
            next: None,
            op: Op::Add,
        };

        ret
    }
}
