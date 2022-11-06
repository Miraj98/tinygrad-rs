use ndarray::{Array2, ShapeBuilder, Dim, arr2};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use std::{cell::RefCell, collections::HashSet};

#[derive(Debug)]
pub struct Tensor<'a> {
    pub data: Array2<f32>,
    pub grad: RefCell<Option<Array2<f32>>>,
    _ctx: Option<OpCtx<'a>>,
}

impl<'a> Tensor<'a> {
    pub fn new(a: Array2<f32>) -> Tensor<'a> {
        Tensor {
            data: a,
            grad: RefCell::new(None),
            _ctx: None,
        }
    }

    pub fn ones<Sh>(s: Sh) -> Tensor<'a>
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            data: Array2::ones(s),
            grad: RefCell::new(None),
            _ctx: None,
        }
    }

    pub fn zeros<Sh>(s: Sh) -> Tensor<'a>
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            data: Array2::zeros(s),
            grad: RefCell::new(None),
            _ctx: None,
        }
    }

    pub fn randn<Sh>(s: Sh) -> Tensor<'a>
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            data: Array2::random(s, StandardNormal),
            grad: RefCell::new(None),
            _ctx: None,
        }
    }
}

impl<'a> Tensor<'a> {
    pub fn add(&'a self, x: &'a Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data + &x.data,
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self), RefCell::new(x)],
                op_type: OpType::Add,
            }),
        }
    }

    pub fn mul(&'a self, x: &'a Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data * &x.data,
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self), RefCell::new(x)],
                op_type: OpType::Mul,
            }),
        }
    }

    pub fn sub(&'a self, x: &'a Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data - &x.data,
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self), RefCell::new(x)],
                op_type: OpType::Sub,
            }),
        }
    }

    pub fn matmul(&'a self, x: &'a Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: (&self.data).dot(&x.data),
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self), RefCell::new(x)],
                op_type: OpType::Matmul,
            }),
        }
    }

    pub fn mean(&'a self) -> Tensor<'a> {
        let mean_val = self.data.mean().unwrap();
        Tensor {
            data: arr2(&[[mean_val]]),
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self)],
                op_type: OpType::Mean,
            }),
        }
    }

    pub fn sum(&'a self) -> Tensor<'a> {
        let sum_val = self.data.mean().unwrap();
        Tensor {
            data: arr2(&[[sum_val]]),
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self)],
                op_type: OpType::Sum,
            }),
        }
    }

    pub fn sigmoid(&'a self) -> Tensor<'a> {
        Tensor {
            data: self.data.mapv(|val| 1.0 / (1.0 + f32::exp(-val))),
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self)],
                op_type: OpType::Sigmoid,
            }),
        }
    }

    pub fn sq(&'a self) -> Tensor<'a> {
        Tensor {
            data: self.data.mapv(|val| val * val),
            grad: RefCell::new(None),
            _ctx: Some(OpCtx {
                inputs: vec![RefCell::new(self)],
                op_type: OpType::Square,
            }),
        }
    }

    pub fn backward(&self) {
        let mut visited = HashSet::<*const Tensor>::new();
        let mut added = HashSet::<*const Tensor>::new();
        let mut topo = Vec::<&Tensor>::new();
        let mut work_stack = Vec::<&Tensor>::new();

        work_stack.push(self);

        while work_stack.len() > 0 {
            let mapped: Vec<&Array2<f32>> =
                work_stack.as_slice().iter().map(|t0| &(*t0).data).collect();
            println!("Current working stack\n{:?}", mapped);

            if let Some(t) = work_stack.pop() {
                println!("Current task\n{:?}", t.data);
                let t_ptr = t as *const Tensor;
                if visited.contains(&t_ptr) {
                    if !added.contains(&t_ptr) {
                        topo.push(t);
                        added.insert(t_ptr);
                    }
                } else {
                    visited.insert(t_ptr);
                    if let Some(ctx) = &t._ctx {
                        if ctx.inputs.len() == 0 {
                            if !added.contains(&t_ptr) {
                                topo.push(t);
                                added.insert(t_ptr);
                            }
                        } else {
                            work_stack.push(t);
                            for _i in &ctx.inputs {
                                let _t = *(*_i).borrow();
                                let _t_ptr = _t as *const Tensor;
                                if !visited.contains(&_t_ptr) {
                                    work_stack.push(_t);
                                }
                            }
                        }
                    } else {
                        println!("Pushing (ctx = None)...\n{:?}", t.data);
                        if !added.contains(&t_ptr) {
                            topo.push(t);
                            added.insert(t_ptr);
                        }
                    };
                }
                println!("\n\n")
            }
        }

        topo.reverse();

        println!("Topo order");
        for elem in topo.iter() {
            println!("{:?}", elem.data);
        }

        let arr = Array2::<f32>::ones(self.data.dim());
        for t in topo {
            match &t._ctx {
                Some(op_node) => {
                    op_node.backward(&arr);
                }
                None => {}
            }
        }
        let mut g = self.grad.borrow_mut();
        *g = Some(arr);
    }
}

#[derive(Debug)]
enum OpType {
    Add,
    Sub,
    Mul,
    Matmul,
    Mean,
    Square,
    Sum,
    Sigmoid
}

#[derive(Debug)]
struct OpCtx<'a> {
    inputs: Vec<RefCell<&'a Tensor<'a>>>,
    op_type: OpType,
}

impl<'a> OpCtx<'a> {
    pub fn backward(&self, incoming_grad: &Array2<f32>) {
        match self.op_type {
            OpType::Add => {
                for i in 0..2 {
                    let mut ref_grad = self.inputs[i].borrow().grad.borrow_mut();
                    let dref_grad = ref_grad.as_ref();
                    match dref_grad {
                        Some(grad) => {
                            *ref_grad = Some(grad + incoming_grad);
                        }
                        None => {
                            *ref_grad = Some(incoming_grad.to_owned());
                        }
                    }
                }
            }
            OpType::Sub => {
                for i in 0..2 {
                    let mut ref_grad = self.inputs[i].borrow().grad.borrow_mut();
                    let dref_grad = ref_grad.as_ref();
                    match dref_grad {
                        Some(grad) => {
                            *ref_grad = Some(grad - incoming_grad);
                        }
                        None => {
                            *ref_grad = Some(-incoming_grad.to_owned());
                        }
                    };
                }
            }
            OpType::Mul => {
                let mut ref_grad_0 = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad_0 = ref_grad_0.as_ref();
                match dref_grad_0 {
                    Some(g) => {
                        *ref_grad_0 = Some(g + incoming_grad * &self.inputs[1].borrow().data);
                    }
                    None => {
                        *ref_grad_0 = Some(incoming_grad * &self.inputs[1].borrow().data);
                    }
                };

                let mut ref_grad_1 = self.inputs[1].borrow().grad.borrow_mut();
                let dref_grad_1 = ref_grad_1.as_ref();
                match dref_grad_1 {
                    Some(g) => {
                        *ref_grad_1 = Some(g + incoming_grad * &self.inputs[0].borrow().data);
                    }
                    None => {
                        *ref_grad_1 = Some(incoming_grad * &self.inputs[0].borrow().data);
                    }
                };
            }
            OpType::Matmul => {
                let mut ref_grad_0 = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad_0 = ref_grad_0.as_ref();
                match dref_grad_0 {
                    Some(g) => {
                        *ref_grad_0 =
                            Some(g + incoming_grad.dot(&self.inputs[1].borrow().data.t()));
                    }
                    None => {
                        *ref_grad_0 = Some(incoming_grad.dot(&self.inputs[1].borrow().data.t()));
                    }
                };

                let mut ref_grad_1 = self.inputs[1].borrow().grad.borrow_mut();
                let dref_grad_1 = ref_grad_1.as_ref();
                match dref_grad_1 {
                    Some(g) => {
                        *ref_grad_1 =
                            Some(g + &self.inputs[0].borrow().data.t().dot(incoming_grad));
                    }
                    None => {
                        *ref_grad_1 = Some(self.inputs[0].borrow().data.t().dot(incoming_grad));
                    }
                };
            }
            OpType::Sigmoid => {
                let mut ref_grad = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad = ref_grad.as_ref();
                match dref_grad {
                    Some(g) => {
                        *ref_grad = Some(g + incoming_grad * &self.inputs[0].borrow().data.mapv(|f| f * (1. - f)));
                    }
                    None => {
                        *ref_grad = Some(incoming_grad * self.inputs[0].borrow().data.mapv(|f| f * (1. - f)));
                    }
                };
            }
            OpType::Mean => {
                let mut ref_grad = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad = ref_grad.as_ref();
                match dref_grad {
                    Some(g) => {
                        let input = self.inputs[0].borrow();
                        *ref_grad = Some(g + incoming_grad / input.data.len() as f32);
                    }
                    None => {
                        let input = self.inputs[0].borrow();
                        *ref_grad = Some(incoming_grad / input.data.len() as f32);
                    }
                };
            }
            OpType::Sum => {
                let mut ref_grad = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad = ref_grad.as_ref();
                match dref_grad {
                    Some(g) => {
                        *ref_grad = Some(g + incoming_grad);
                    }
                    None => {
                        *ref_grad = Some(incoming_grad.to_owned());
                    }
                };
            }
            OpType::Square => {
                let mut ref_grad = self.inputs[0].borrow().grad.borrow_mut();
                let dref_grad = ref_grad.as_ref();
                match dref_grad {
                    Some(g) => {
                        *ref_grad = Some(g + 2.0 * &self.inputs[0].borrow().data * incoming_grad);
                    }
                    None => {
                        *ref_grad = Some(2.0 * incoming_grad);
                    }
                };
            }
        };
    }
}
