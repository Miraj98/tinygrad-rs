pub mod ops;

use ndarray::{arr2, Array2, Dim, ShapeBuilder};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use ops::{
    Backprop, BinaryOpType, BinaryOps, OpType, ReduceOpTypes, ReduceOps, TensorConstructors,
    UnaryOpType, UnaryOps,
};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Debug)]
pub struct TensorData {
    pub value: RefCell<Array2<f64>>,
    pub grad: RefCell<Option<Array2<f64>>>,
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
}

pub type Tensor = Rc<TensorCore>;

impl TensorConstructors for TensorCore {
    fn new(a: Array2<f64>) -> Tensor {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(a),
                grad: RefCell::new(None),
            },
            ctx: None,
        })
    }

    fn ones<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::ones(shape)),
                grad: RefCell::new(None),
            },
            ctx: None,
        })
    }

    fn zeros<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::zeros(shape)),
                grad: RefCell::new(None),
            },
            ctx: None,
        })
    }

    fn fill<Sh>(shape: Sh, x: f64) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::ones(shape) * x),
                grad: RefCell::new(None),
            },
            ctx: None,
        })
    }

    fn randn<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(Array2::random(shape, StandardNormal)),
                grad: RefCell::new(None),
            },
            ctx: None,
        })
    }
}

impl BinaryOps for Tensor {
    fn add(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s + _x),
                grad: RefCell::new(None),
            },
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Add),
            }),
        })
    }

    fn sub(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s - _x),
                grad: RefCell::new(None),
            },
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Sub),
            }),
        })
    }

    fn mul(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s * _x),
                grad: RefCell::new(None),
            },
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Mul),
            }),
        })
    }

    fn matmul(&self, x: &Tensor) -> Tensor {
        let _s = &self.data.value.borrow() as &Array2<f64>;
        let _x = &x.data.value.borrow() as &Array2<f64>;

        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(_s.dot(_x)),
                grad: RefCell::new(None),
            },
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Matmul),
            }),
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
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self)],
                op_type: OpType::UnaryOp(UnaryOpType::Sigmoid),
            }),
        })
    }

    fn square(&self) -> Tensor {
        let a = self.data.value.borrow().mapv(|val| val * val);
        Rc::new(TensorCore {
            data: TensorData {
                value: RefCell::new(a),
                grad: RefCell::new(None),
            },
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self)],
                op_type: OpType::UnaryOp(UnaryOpType::Square),
            }),
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
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self)],
                op_type: OpType::ReduceOp(ReduceOpTypes::Mean),
            }),
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
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self)],
                op_type: OpType::ReduceOp(ReduceOpTypes::Sum),
            }),
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

        let arr = Array2::<f64>::ones(self.data.value.borrow().dim());
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
    fn _backward(&self, incoming_grad: &Array2<f64>) {
        if let Some(ctx) = &self.ctx {
            match ctx.op_type {
                OpType::BinaryOp(BinaryOpType::Add) => {
                    for i in 0..2 {
                        let mut t = ctx.saved_tensors[i].as_ref().data.grad.borrow_mut();
                        let grad_optn = t.as_ref();
                        if let Some(g) = grad_optn {
                            *t = Some(g + incoming_grad);
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

                    if let Some(g) = grad_optn0 {
                        *t0 = Some(
                            g + incoming_grad
                                .dot(&ctx.saved_tensors[1].as_ref().data.value.borrow().t()),
                        );
                    } else {
                        *t0 = Some(
                            incoming_grad
                                .dot(&ctx.saved_tensors[1].as_ref().data.value.borrow().t()),
                        );
                    }

                    if let Some(g) = grad_optn1 {
                        *t1 = Some(
                            g + ctx.saved_tensors[0]
                                .as_ref()
                                .data
                                .value
                                .borrow()
                                .t()
                                .dot(incoming_grad),
                        );
                    } else {
                        *t1 = Some(
                            ctx.saved_tensors[0]
                                .as_ref()
                                .data
                                .value
                                .borrow()
                                .t()
                                .dot(incoming_grad),
                        );
                    }
                }
                OpType::BinaryOp(BinaryOpType::Mul) => {
                    let mut t0 = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let mut t1 = ctx.saved_tensors[1].as_ref().data.grad.borrow_mut();
                    let grad_optn0 = t0.as_ref();
                    let grad_optn1 = t1.as_ref();

                    if let Some(g) = grad_optn0 {
                        *t0 = Some(
                            g + incoming_grad
                                * &ctx.saved_tensors[1].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    } else {
                        *t0 = Some(
                            incoming_grad
                                * &ctx.saved_tensors[1].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    }

                    if let Some(g) = grad_optn1 {
                        *t1 = Some(
                            g + incoming_grad
                                * &ctx.saved_tensors[0].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    } else {
                        *t1 = Some(
                            incoming_grad
                                * &ctx.saved_tensors[0].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    }
                }
                OpType::BinaryOp(BinaryOpType::Sub) => {
                    for i in 0..2 {
                        let mut t = ctx.saved_tensors[i].as_ref().data.grad.borrow_mut();
                        let grad_optn = t.as_ref();
                        if let Some(g) = grad_optn {
                            *t = Some(g - incoming_grad);
                        } else {
                            *t = Some((-incoming_grad).to_owned());
                        }
                    }
                }
                OpType::UnaryOp(UnaryOpType::Sigmoid) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(
                            g + incoming_grad
                                * &ctx.saved_tensors[0]
                                    .as_ref()
                                    .data
                                    .value
                                    .borrow()
                                    .mapv(|f| f * (1. - f)),
                        );
                    } else {
                        *t = Some(
                            incoming_grad
                                * &ctx.saved_tensors[0]
                                    .as_ref()
                                    .data
                                    .value
                                    .borrow()
                                    .mapv(|f| f * (1. - f)),
                        );
                    }
                }
                OpType::ReduceOp(ReduceOpTypes::Mean) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(
                            g + incoming_grad
                                / ctx.saved_tensors[0].as_ref().data.value.borrow().len() as f64,
                        );
                    } else {
                        *t = Some(
                            incoming_grad
                                / ctx.saved_tensors[0].as_ref().data.value.borrow().len() as f64,
                        );
                    }
                }
                OpType::UnaryOp(UnaryOpType::Square) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(
                            g + incoming_grad
                                * 2.0
                                * &ctx.saved_tensors[0].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    } else {
                        *t = Some(
                            incoming_grad
                                * 2.0
                                * &ctx.saved_tensors[0].as_ref().data.value.borrow()
                                    as &Array2<f64>,
                        );
                    }
                }
                OpType::ReduceOp(ReduceOpTypes::Sum) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(g + incoming_grad);
                    } else {
                        *t = Some(incoming_grad.to_owned());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, array, Array2};
    use super::{
        ops::{BinaryOps, TensorConstructors},
        TensorCore,
    };

    #[test]
    fn matmul_test() {
        let a = arr2(&[[2.0, 3.0], [4.0, 5.0]]);
        let b = arr2(&[[15.0, 8.0], [10.0, 71.0]]);

        let a_tensor = TensorCore::new(a);
        let b_tensor = TensorCore::new(b);
        let c = a_tensor.matmul(&b_tensor);
        let d = &c.data.value.borrow() as &Array2<f64>;

        assert_eq!(d, array![[60., 229.], [110., 387.]]);
        assert_eq!(c.ctx.as_ref().unwrap().saved_tensors.len(), 2);
        assert_eq!(&c.ctx.as_ref().unwrap().saved_tensors[0].data.value.borrow() as &Array2<f64>, array![[2.0, 3.0], [4.0, 5.0]]);
        assert_eq!(&c.ctx.as_ref().unwrap().saved_tensors[1].data.value.borrow() as &Array2<f64>, array![[15.0, 8.0], [10.0, 71.0]]);
    }
}
