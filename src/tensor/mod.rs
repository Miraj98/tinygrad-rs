pub mod ops;

use ndarray::{Array2, Dim, ShapeBuilder};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use ops::{Backprop, BinaryOpType, BinaryOps, OpType, TensorConstructors, UnaryOpType, UnaryOps};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Debug)]
struct TensorData {
    pub value: Array2<f32>,
    pub grad: RefCell<Option<Array2<f32>>>,
}

#[derive(Debug)]
struct TensorContext {
    saved_tensors: Vec<Rc<TensorCore>>,
    op_type: OpType,
}

#[derive(Debug)]
pub struct TensorCore {
    data: Rc<TensorData>,
    ctx: Option<TensorContext>,
}

pub type Tensor = Rc<TensorCore>;

impl TensorConstructors for TensorCore {
    fn new(a: Array2<f32>) -> Tensor {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: a,
                grad: RefCell::new(None),
            }),
            ctx: None,
        })
    }

    fn ones<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: Array2::ones(shape),
                grad: RefCell::new(None),
            }),
            ctx: None,
        })
    }

    fn zeros<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: Array2::zeros(shape),
                grad: RefCell::new(None),
            }),
            ctx: None,
        })
    }

    fn fill<Sh>(shape: Sh, x: f32) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: Array2::ones(shape) * x,
                grad: RefCell::new(None),
            }),
            ctx: None,
        })
    }

    fn randn<Sh>(shape: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: Array2::random(shape, StandardNormal),
                grad: RefCell::new(None),
            }),
            ctx: None,
        })
    }
}

impl BinaryOps for Tensor {
    fn add(&self, x: &Tensor) -> Tensor {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: &self.data.value + &x.data.value,
                grad: RefCell::new(None),
            }),
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Add),
            }),
        })
    }

    fn sub(&self, x: &Tensor) -> Tensor {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: &self.data.value - &x.data.value,
                grad: RefCell::new(None),
            }),
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Sub),
            }),
        })
    }

    fn mul(&self, x: &Tensor) -> Tensor {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: &self.data.value * &x.data.value,
                grad: RefCell::new(None),
            }),
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Mul),
            }),
        })
    }

    fn matmul(&self, x: &Tensor) -> Tensor {
        Rc::new(TensorCore {
            data: Rc::new(TensorData {
                value: (&self.data.value).dot(&x.data.value),
                grad: RefCell::new(None),
            }),
            ctx: Some(TensorContext {
                saved_tensors: vec![Rc::clone(self), Rc::clone(x)],
                op_type: OpType::BinaryOp(BinaryOpType::Matmul),
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

        let arr = Array2::<f32>::ones(self.data.value.dim());
        for i in &topo {
            unsafe {
                match (**i).data.as_ref().grad.borrow().as_ref() {
                    Some(g) => {
                        (**i)._backward(g)
                    }
                    None => {
                        (**i)._backward(&arr)
                    }
                }
            }
            unsafe {
                println!("{:?}", (**i).data.value);
            }
        }

        println!("{:?}", topo);
    }
}

impl TensorCore {
    fn _backward(&self, incoming_grad: &Array2<f32>) {
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
                            g + incoming_grad.dot(&ctx.saved_tensors[1].as_ref().data.value.t()),
                        );
                    } else {
                        *t0 =
                            Some(incoming_grad.dot(&ctx.saved_tensors[1].as_ref().data.value.t()));
                    }

                    if let Some(g) = grad_optn1 {
                        *t1 = Some(
                            g + ctx.saved_tensors[1]
                                .as_ref()
                                .data
                                .value
                                .t()
                                .dot(incoming_grad),
                        );
                    } else {
                        *t1 = Some(
                            ctx.saved_tensors[1]
                                .as_ref()
                                .data
                                .value
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
                        *t0 = Some(g + incoming_grad * &ctx.saved_tensors[1].as_ref().data.value);
                    } else {
                        *t0 = Some(incoming_grad * &ctx.saved_tensors[1].as_ref().data.value);
                    }

                    if let Some(g) = grad_optn1 {
                        *t1 = Some(g + incoming_grad * &ctx.saved_tensors[0].as_ref().data.value);
                    } else {
                        *t1 = Some(incoming_grad * &ctx.saved_tensors[0].as_ref().data.value);
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
                                    .mapv(|f| f * (1. - f)),
                        );
                    } else {
                        *t = Some(
                            incoming_grad
                                * &ctx.saved_tensors[0]
                                    .as_ref()
                                    .data
                                    .value
                                    .mapv(|f| f * (1. - f)),
                        );
                    }
                }
                OpType::UnaryOp(UnaryOpType::Mean) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(
                            g + incoming_grad
                                / ctx.saved_tensors[0].as_ref().data.value.len() as f32,
                        );
                    } else {
                        *t = Some(
                            incoming_grad / ctx.saved_tensors[0].as_ref().data.value.len() as f32,
                        );
                    }
                }
                OpType::UnaryOp(UnaryOpType::Square) => {
                    let mut t = ctx.saved_tensors[0].as_ref().data.grad.borrow_mut();
                    let grad_optn = t.as_ref();
                    if let Some(g) = grad_optn {
                        *t = Some(
                            g + incoming_grad * 2.0 * &ctx.saved_tensors[0].as_ref().data.value,
                        );
                    } else {
                        *t = Some(incoming_grad * 2.0 * &ctx.saved_tensors[0].as_ref().data.value);
                    }
                }
                OpType::UnaryOp(UnaryOpType::Sum) => {
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
