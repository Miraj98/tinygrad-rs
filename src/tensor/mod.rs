use ndarray::{Array2, ShapeBuilder, Dim, arr2};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use std::{cell::RefCell, collections::HashSet, rc::Weak};

pub struct Tensor {
    t: RefCell<TensorData>,
}

#[derive(Debug)]
pub struct TensorData {
    pub data: Array2<f32>,
    pub grad: Option<Array2<f32>>,
    saved_tensors: Option<Vec<Weak<RefCell<TensorData>>>>,
    op_type: Option<OpType>,
    //._ctx: Option<OpCtx<'a>>,
}

impl Tensor {
    pub fn new(a: Array2<f32>) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: a,
                grad: None,
                saved_tensors: None,
                op_type: None,
            }),
        }
    }

    pub fn ones<Sh>(s: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            t: RefCell::new(TensorData {
                data: Array2::ones(s),
                grad: None,
                saved_tensors: None,
                op_type: None
            }),
        }
    }

    pub fn zeros<Sh>(s: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            t: RefCell::new(TensorData {
                data: Array2::zeros(s),
                grad: None,
                saved_tensors: None,
                op_type: None
            }),
        }
    }

    pub fn randn<Sh>(s: Sh) -> Tensor
    where
        Sh: ShapeBuilder<Dim = Dim<[usize; 2]>>,
    {
        Tensor {
            t: RefCell::new(TensorData {
                data: Array2::random(s, StandardNormal),
                grad: None,
                saved_tensors: None,
                op_type: None,
            }),
        }
    }
}

impl Tensor {
    pub fn add(&self, x: & Tensor) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: &self.data + &x.data,
                grad: None,
                saved_tensors: Some(vec![Weak::from(RefCell::new(self)), Weak::from(RefCell::new(x))]),
                op_type: Some(OpType::Add),
            }),
        }
    }

    pub fn mul(&self, x: &Tensor) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: &self.data * &x.data,
                grad: None,
                saved_tensors: Some(vec![Weak::from(RefCell::new(self)), Weak::from(RefCell::new(x))]),
                op_type: Some(OpType::Mul),
             }),
        }
    }

    pub fn sub(&self, x: &Tensor) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: &self.data - &x.data,
                grad: RefCell::new(None),
                saved_tensors: Some(vec![Weak::from(RefCell::new(self)), Weak::from(RefCell::new(x))]),
                op_type: Some(OpType::Sub),
            })
        }
    }

    pub fn matmul(&self, x: &Tensor) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: (&self.data).dot(&x.data),
                grad: None,
                saved_tensors: Some(vec![Weak::from(RefCell::new(self)), Weak::from(RefCell::new(x))]),
                op_type: Some(OpType::Matmul),
            })
        }
    }

    pub fn mean(&self) -> Tensor {
        let mean_val = self.data.mean().unwrap();
        Tensor {
            t: RefCell::new(TensorData {
                data: arr2(&[[mean_val]]),
                grad: (None),
                saved_tensors: Some(vec![Weak::from(RefCell::new(self))]),
                op_type: Some(OpType::Mean),
            })
        }
    }

    pub fn sum(&self) -> Tensor {
        let sum_val = self.data.mean().unwrap();
        Tensor {
            t: RefCell::new(TensorData {
                data: arr2(&[[sum_val]]),
                grad: (None),
                saved_tensors: Some(vec![Weak::from(RefCell::new(self))]),
                op_type: Some(OpType::Sum),
            })
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: self.data.mapv(|val| 1.0 / (1.0 + f32::exp(-val))),
                grad: (None),
                saved_tensors: Some(vec![Weak::from(RefCell::new(self))]),
                op_type: Some(OpType::Sigmoid),
            })
        }
    }

    pub fn sq(&self) -> Tensor {
        Tensor {
            t: RefCell::new(TensorData {
                data: self.t.borrow().data.mapv(|val| val * val),
                grad: (None),
                saved_tensors: Some(vec![Weak::from(RefCell::new(self.t.borrow()))]),
                op_type: Some(OpType::Square),
            })
        }
    }

    pub fn backward(&self) {
        let mut visited = HashSet::<*const Tensor>::new();
        let mut added = HashSet::<*const Tensor>::new();
        let mut topo = Vec::<&Tensor>::new();
        let mut work_stack = Vec::<&Tensor>::new();

        work_stack.push(self);

        while work_stack.len() > 0 {
            if let Some(t) = work_stack.pop() {
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
                        if !added.contains(&t_ptr) {
                            topo.push(t);
                            added.insert(t_ptr);
                        }
                    };
                }
            }
        }

        topo.reverse();

        let arr = Array2::<f32>::ones(self.data.dim());
        for t in topo {
            match &t._ctx {
                Some(op_node) => {
                    if let Some(grad) = &*t.grad.borrow() {
                        op_node.backward(&grad);
                    } else {
                        op_node.backward(&arr);
                    }
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
