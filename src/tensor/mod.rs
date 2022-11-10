pub mod ops;

use ndarray::{arr2, Array2, Dim, Dimension, Ix2, ShapeBuilder};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use ops::{
    Backprop, BinaryOpType, BinaryOps, OpType, ReduceOpTypes, ReduceOps, TensorConstructors,
    UnaryOpType, UnaryOps,
};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Debug)]
pub struct TensorData {
    pub value: RefCell<Array2<f64>>,
    pub grad: RefCell<Option<Tensor>>,
}

#[derive(Debug)]
pub struct TensorContext {
    pub saved_tensors: Vec<Rc<TensorCore>>,
    pub op_type: OpType,
}

#[derive(Debug)]
pub struct TensorCore {
    pub data: TensorData,
    pub ctx: Option<TensorContext>,
    pub requires_grad: bool,
}

pub type Tensor = Rc<TensorCore>;

impl TensorConstructors for TensorCore {
    fn new(a: Array2<f64>, requires_grad: bool) -> Tensor {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(a),
                grad: RefCell::new(None),
            },
            ctx: None,
            requires_grad,
        })
    }

    fn ones<Sh>(shape: Sh, requires_grad: bool) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::ones(shape)),
                grad: RefCell::new(None),
            },
            ctx: None,
            requires_grad,
        })
    }

    fn zeros<Sh>(shape: Sh, requires_grad: bool) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::zeros(shape)),
                grad: RefCell::new(None),
            },
            ctx: None,
            requires_grad,
        })
    }

    fn fill<Sh>(shape: Sh, x: f64, requires_grad: bool) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::ones(shape) * x),
                grad: RefCell::new(None),
            },
            ctx: None,
            requires_grad,
        })
    }

    fn randn<Sh>(shape: Sh, requires_grad: bool) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::random(shape, StandardNormal)),
                grad: RefCell::new(None),
            },
            ctx: None,
            requires_grad,
        })
    }
}

impl BinaryOps for Tensor {
    fn add(&self, x: &Tensor) -> Tensor {
        let _s = self.ndarray();
        let _x = x.ndarray();
        let requires_grad = self.requires_grad || x.requires_grad;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s + _x),
                grad: RefCell::new(None),
            },
            ctx: if requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                    op_type: OpType::BinaryOp(BinaryOpType::Add),
                })
            } else {
                None
            },
            requires_grad,
        })
    }

    fn sub(&self, x: &Tensor) -> Tensor {
        let _s = self.ndarray();
        let _x = x.ndarray();
        let requires_grad = self.requires_grad || x.requires_grad;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s - _x),
                grad: RefCell::new(None),
            },
            ctx: if requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                    op_type: OpType::BinaryOp(BinaryOpType::Sub),
                })
            } else {
                None
            },
            requires_grad,
        })
    }

    fn mul(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;
        let requires_grad = self.requires_grad || x.requires_grad;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s * _x),
                grad: RefCell::new(None),
            },
            ctx: if requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                    op_type: OpType::BinaryOp(BinaryOpType::Mul),
                })
            } else {
                None
            },
            requires_grad,
        })
    }

    fn matmul(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;
        let requires_grad = self.requires_grad || x.requires_grad;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s.dot(_x)),
                grad: RefCell::new(None),
            },
            ctx: if requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                    op_type: OpType::BinaryOp(BinaryOpType::Matmul),
                })
            } else {
                None
            },
            requires_grad,
        })
    }
}

impl UnaryOps for Tensor {
    fn sigmoid(&self) -> Tensor {
        let a = self
            .data
            .value
            .borrow()
            .mapv(|val| 1.0 / (1.0 + f64::exp(-val)));
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(a),
                grad: RefCell::new(None),
            },
            ctx: if self.requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self)],
                    op_type: OpType::UnaryOp(UnaryOpType::Sigmoid),
                })
            } else {
                None
            },
            requires_grad: self.requires_grad,
        })
    }

    fn square(&self) -> Tensor {
        let a = self.data.value.borrow().mapv(|val| val * val);
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(a),
                grad: RefCell::new(None),
            },
            ctx: if self.requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self)],
                    op_type: OpType::UnaryOp(UnaryOpType::Square),
                })
            } else {
                None
            },
            requires_grad: self.requires_grad,
        })
    }
}

impl ReduceOps for Tensor {
    fn mean(&self) -> Tensor {
        let a = self.data.value.borrow().mean().unwrap();
        let array2d = arr2(&[[a]]);
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(array2d),
                grad: RefCell::new(None),
            },
            ctx: if self.requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self)],
                    op_type: OpType::ReduceOp(ReduceOpTypes::Mean),
                })
            } else {
                None
            },
            requires_grad: self.requires_grad,
        })
    }

    fn sum(&self) -> Tensor {
        let a = self.data.value.borrow().sum();
        let array2d = arr2(&[[a]]);
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(array2d),
                grad: RefCell::new(None),
            },
            ctx: if self.requires_grad {
                Some(TensorContext {
                    saved_tensors: vec![Rc::clone(self)],
                    op_type: OpType::ReduceOp(ReduceOpTypes::Sum),
                })
            } else {
                None
            },
            requires_grad: self.requires_grad,
        })
    }
}

impl Backprop for Tensor {
    fn backward(&self) {
        let mut visited = HashSet::<*const TensorCore>::new();
        let mut added = HashSet::<*const TensorCore>::new();
        let mut work_stack = Vec::<*const TensorCore>::new();

        let mut topo = Vec::<*const TensorCore>::new();

        work_stack.push(self.as_ref());

        while work_stack.len() > 0 {
            if let Some(t) = work_stack.pop() {
                if visited.contains(&t) {
                    if !added.contains(&t) {
                        topo.push(t);
                        added.insert(t);
                    }
                } else {
                    visited.insert(t);
                    unsafe {
                        if let Some(ctx) = &(*t).ctx {
                            if ctx.saved_tensors.len() == 0 {
                                if !added.contains(&t) {
                                    topo.push(t);
                                    added.insert(t);
                                }
                            } else {
                                work_stack.push(t);
                                for st in &ctx.saved_tensors {
                                    let st_ptr = st.as_ref() as *const TensorCore;
                                    if !visited.contains(&st_ptr) {
                                        work_stack.push(st_ptr);
                                    }
                                }
                            }
                        } else {
                            if !added.contains(&t) {
                                topo.push(t);
                                added.insert(t);
                            }
                        };
                    }
                }
            }
        }

        topo.reverse();

        let arr = TensorCore::ones(self.dim(), false);
        for i in &topo {
            unsafe {
                match (**i).data.grad.borrow().as_ref() {
                    Some(g) => (**i)._backward(g),
                    None => (**i)._backward(&arr),
                }
            }
        }

        let mut g = self.data.grad.borrow_mut();
        *g = Some(arr);
    }

    fn zero_grad(&self) {
        let mut visited = HashSet::<*const TensorCore>::new();
        let mut added = HashSet::<*const TensorCore>::new();
        let mut work_stack = Vec::<*const TensorCore>::new();

        let mut topo = Vec::<*const TensorCore>::new();

        work_stack.push(self.as_ref());

        while work_stack.len() > 0 {
            if let Some(t) = work_stack.pop() {
                if visited.contains(&t) {
                    if !added.contains(&t) {
                        topo.push(t);
                        added.insert(t);
                    }
                } else {
                    visited.insert(t);
                    unsafe {
                        if let Some(ctx) = &(*t).ctx {
                            if ctx.saved_tensors.len() == 0 {
                                if !added.contains(&t) {
                                    topo.push(t);
                                    added.insert(t);
                                }
                            } else {
                                work_stack.push(t);
                                for st in &ctx.saved_tensors {
                                    let st_ptr = st.as_ref() as *const TensorCore;
                                    if !visited.contains(&st_ptr) {
                                        work_stack.push(st_ptr);
                                    }
                                }
                            }
                        } else {
                            if !added.contains(&t) {
                                topo.push(t);
                                added.insert(t);
                            }
                        };
                    }
                }
            }
        }

        for i in &topo {
            unsafe {
                let mut g = (**i).data.grad.borrow_mut();
                *g = None;
            }
        }
    }
}

impl TensorCore {
    fn _backward(&self, incoming_grad: &Tensor) {
        if let Some(ctx) = &self.ctx {
            match &ctx.op_type {
                OpType::BinaryOp(BinaryOpType::Add) => {
                    for i in 0..2 {
                        let mut t = ctx.saved_tensors[i].as_ref().data.grad.borrow_mut();
                        let grad_optn = t.as_ref();
                        if let Some(g) = grad_optn {
                            *t = Some(g.add(incoming_grad));
                        } else {
                            *t = Some(incoming_grad.to_owned());
                        }
                    }
                }
                OpType::BinaryOp(BinaryOpType::Matmul) => {
                    let mut t0 = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let mut t1 = ctx.saved_tensors[1].as_ref().data.grad.borrow_mut();
                    let grad_optn0 = t0.as_ref();
                    let grad_optn1 = t1.as_ref();
                    let local_grad0 = incoming_grad
                        .ndarray()
                        .dot(&ctx.saved_tensors[1].as_ref().ndarray().t());
                    let local_grad_tensor0 = TensorCore::new(local_grad0, false);

                    if let Some(g) = grad_optn0 {
                        *t0 = Some(g.add(&local_grad_tensor0));
                    } else {
                        *t0 = Some(local_grad_tensor0);
                    }

                    let local_grad1 = ctx.saved_tensors[0]
                        .as_ref()
                        .ndarray()
                        .t()
                        .dot(incoming_grad.ndarray());
                    let local_grad_tensor1 = TensorCore::new(local_grad1, false);
                    if let Some(g) = grad_optn1 {
                        *t1 = Some(g.add(&local_grad_tensor1));
                    } else {
                        *t1 = Some(local_grad_tensor1);
                    }
                }
                OpType::BinaryOp(BinaryOpType::Mul) => {
                    let mut t0 = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let mut t1 = ctx.saved_tensors[1].as_ref().data.grad.borrow_mut();
                    let grad_optn0 = t0.as_ref();
                    let grad_optn1 = t1.as_ref();

                    if let Some(g) = grad_optn0 {
                        *t0 = Some(incoming_grad.mul(&ctx.saved_tensors[1]).add(&g));
                    } else {
                        *t0 = Some(incoming_grad.mul(&ctx.saved_tensors[1]));
                    }

                    if let Some(g) = grad_optn1 {
                        *t1 = Some(incoming_grad.mul(&ctx.saved_tensors[0]).add(&g));
                    } else {
                        *t1 = Some(incoming_grad.mul(&ctx.saved_tensors[0]));
                    }
                }
                OpType::BinaryOp(BinaryOpType::Sub) => {
                    for i in 0..2 {
                        let mut t = ctx.saved_tensors[i].as_ref().data.grad.borrow_mut();
                        let grad_optn = t.as_ref();
                        if let Some(g) = grad_optn {
                            *t = Some(g.sub(&incoming_grad));
                        } else {
                            *t = Some(incoming_grad.mul(&TensorCore::fill(
                                incoming_grad.dim(),
                                -1.0,
                                false,
                            )));
                        }
                    }
                }
                OpType::UnaryOp(UnaryOpType::Sigmoid) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    let d_sigmoid = TensorCore::new(
                        ctx.saved_tensors[0].ndarray().mapv(|f| f * (1. - f)),
                        false,
                    );
                    if let Some(g) = grad_optn {
                        *t = Some(incoming_grad.mul(&d_sigmoid).add(&g));
                    } else {
                        *t = Some(incoming_grad.mul(&d_sigmoid));
                    }
                }
                OpType::ReduceOp(reduce_op) => {
                    let mut i_g = Array2::<f64>::ones(ctx.saved_tensors[0].as_ref().dim())
                        * incoming_grad.ndarray()[(0, 0)];
                    match reduce_op {
                        ReduceOpTypes::Mean => {
                            i_g = i_g / ctx.saved_tensors[0].as_ref().ndarray().len() as f64;
                            let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                            let grad_optn = t.as_ref();
                            if let Some(g) = grad_optn {
                                *t = Some(TensorCore::new(i_g, false).add(&g));
                            } else {
                                *t = Some(TensorCore::new(i_g, false));
                            }
                        }
                        ReduceOpTypes::Sum => {
                            let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                            let grad_optn = t.as_ref();
                            if let Some(g) = grad_optn {
                                *t = Some(TensorCore::new(i_g, false).add(&g));
                            } else {
                                *t = Some(TensorCore::new(i_g, false));
                            }
                        }
                    }
                }
                OpType::UnaryOp(UnaryOpType::Square) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    let local_grad =
                        incoming_grad.ndarray() * 2.0 * ctx.saved_tensors[0].as_ref().ndarray();
                    let local_grad_tensor = TensorCore::new(local_grad, false);

                    if let Some(g) = grad_optn {
                        *t = Some(local_grad_tensor.add(&g));
                    } else {
                        *t = Some(local_grad_tensor);
                    }
                }
            }
        }
    }
}

impl TensorCore {
    pub fn shape(&self) -> &[usize] {
        self.data.value.borrow().shape()
    }

    pub fn dim(&self) -> (usize, usize) {
        self.data.value.borrow().dim()
    }

    pub fn grad(&self) -> Option<&Tensor> {
        self.data.grad.borrow().as_ref()
    }

    pub fn ndarray(&self) -> &Array2<f64> {
        &self.data.value.borrow()
    }

    pub fn saved_tensors(&self) -> Option<&Vec<Tensor>> {
        if let Some(ctx) = self.ctx.as_ref() {
            return Some(&ctx.saved_tensors);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ops::{Backprop, BinaryOps, TensorConstructors},
        TensorCore,
    };
    use crate::tensor::ops::{ReduceOps, UnaryOps};
    use ndarray::{arr2, array, Array2};

    #[test]
    fn matmul_test() {
        let a = arr2(&[[2.0, 3.0], [4.0, 5.0]]);
        let b = arr2(&[[15.0, 8.0], [10.0, 71.0]]);

        let a_tensor = TensorCore::new(a, true);
        let b_tensor = TensorCore::new(b, true);
        let c = a_tensor.matmul(&b_tensor);
        let d = c.ndarray();

        assert_eq!(d, array![[60., 229.], [110., 387.]]);
        assert_eq!(c.saved_tensors().unwrap().len(), 2);
        assert_eq!(
            c.saved_tensors().unwrap()[0].ndarray(),
            array![[2.0, 3.0], [4.0, 5.0]]
        );
        assert_eq!(
            c.saved_tensors().unwrap()[1].ndarray(),
            array![[15.0, 8.0], [10.0, 71.0]]
        );
    }

    #[test]
    fn chain_ops() {
        let t1 = TensorCore::new(arr2(&[[2.0, 3.0], [4.0, 5.0]]), true);
        let t2 = TensorCore::new(arr2(&[[15.0, 8.0], [10.0, 71.0]]), true);
        let t3 = TensorCore::new(arr2(&[[58.0, 220.0], [100.0, 380.0]]), true);

        let l = t1.matmul(&t2).sub(&t3).square().mean();
        let d = l.ndarray();
        assert_eq!(d.len(), 1);
        assert_eq!(d, array![[58.5000]]);
        assert_eq!(l.saved_tensors().unwrap().len(), 1);

        let ls0 = &l.ctx.as_ref().unwrap().saved_tensors[0]; // The tensor with the square value
        assert_eq!(ls0.ctx.as_ref().unwrap().saved_tensors.len(), 1);
        assert_eq!(
            &ls0.data.value.borrow() as &Array2<f64>,
            array![[4., 81.], [100., 49.]]
        );

        let ls00 = &ls0.ctx.as_ref().unwrap().saved_tensors[0]; // The tensor output from the subtract op
        assert_eq!(ls00.ctx.as_ref().unwrap().saved_tensors.len(), 2);
        assert_eq!(
            &ls00.data.value.borrow() as &Array2<f64>,
            array![[2., 9.,], [10., 7.]]
        );

        let ls000 = &ls00.ctx.as_ref().unwrap().saved_tensors[0]; // The tensor output from matmul op
        let ls001 = &ls00.ctx.as_ref().unwrap().saved_tensors[1]; // t3
        assert_eq!(ls000.ctx.as_ref().unwrap().saved_tensors.len(), 2);
        assert_eq!(ls001.ctx.is_none(), true);
        assert_eq!(
            &ls000.data.value.borrow() as &Array2<f64>,
            array![[60., 229.], [110., 387.]]
        );

        let _t1 = &ls000.ctx.as_ref().unwrap().saved_tensors[0];
        let _t2 = &ls000.ctx.as_ref().unwrap().saved_tensors[1];

        // let ls00 = &ls0.ctx.as_ref().unwrap().saved_tensors[0];
        assert_eq!(_t1.ctx.is_none(), true);
        assert_eq!(_t2.ctx.is_none(), true);
        assert_eq!(
            &_t1.data.value.borrow() as &Array2<f64>,
            array![[2.0, 3.0], [4.0, 5.0]]
        );
        assert_eq!(
            &_t2.data.value.borrow() as &Array2<f64>,
            array![[15.0, 8.0], [10.0, 71.0]]
        );
    }

    #[test]
    fn check_grad() {
        let a1 = arr2(&[[2.0, 3.0], [4.0, 5.0]]);
        let a2 = arr2(&[[15.0, 8.0], [10.0, 71.0]]);
        let a3 = arr2(&[[58.0, 220.0], [100.0, 380.0]]);

        let t1 = TensorCore::new(a1.clone(), true);
        let t2 = TensorCore::new(a2.clone(), true);
        let t3 = TensorCore::new(a3.clone(), true);

        let _add = t1.add(&t2);
        _add.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[1., 1.], [1., 1.]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[1., 1.], [1., 1.]]
        );
        _add.zero_grad();

        let _mul = t1.mul(&t2);
        _mul.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[15.0, 8.0], [10.0, 71.0]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[2.0, 3.0], [4.0, 5.0]]
        );
        _mul.zero_grad();

        let _z = t1.matmul(&t2).add(&t3);
        _z.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[23., 81.], [23., 81.]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[6., 6.], [8., 8.]]
        );
        assert_eq!(
            t3.grad().unwrap().ndarray(),
            array![[1., 1.], [1., 1.]]
        );
        _z.zero_grad();

        let _z_reduce_sum = t1.matmul(&t2).add(&t3).sum();
        _z_reduce_sum.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[23., 81.], [23., 81.]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[6., 6.], [8., 8.]]
        );
        assert_eq!(
            t3.grad().unwrap().ndarray(),
            array![[1., 1.], [1., 1.]]
        );
        _z_reduce_sum.zero_grad();

        let _z_reduce_mean = t1.matmul(&t2).add(&t3).mean();
        _z_reduce_mean.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[5.7500, 20.2500], [5.7500, 20.2500]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[1.5000, 1.5000], [2.0000, 2.0000]]
        );
        assert_eq!(
            t3.grad().unwrap().ndarray(),
            array![[0.2500, 0.2500], [0.2500, 0.2500]]
        );
        _z_reduce_mean.zero_grad();

        let _z_reduce_sq = t1.matmul(&t2).add(&t3).square().sum();
        _z_reduce_sq.backward();
        assert_eq!(
            t1.grad().unwrap().ndarray(),
            array![[10724., 66118.], [18572., 113114.]]
        );
        assert_eq!(
            t2.grad().unwrap().ndarray(),
            array![[2152., 7932.], [2808., 10364.]]
        );
        assert_eq!(
            t3.grad().unwrap().ndarray(),
            array![[236., 898.], [420., 1534.]]
        );
        _z_reduce_mean.zero_grad();
    }
}
