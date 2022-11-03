mod ops;

use std::{collections::HashSet, ops::DerefMut};

use ndarray::Array2;

#[derive(Debug)]
enum OpType {
    Add,
    Sub,
    Mul,
}

// a + b = c
// c*e = d
// d.backward()
// dd/dc, dd/de, dd/da

#[derive(Debug)]
struct OpCtx<'a> {
    inputs: Vec<&'a mut Tensor<'a>>,
    op_type: OpType,
}

impl<'a> OpCtx<'a> {
    pub fn backward(&mut self, incoming_grad: &Array2<f64>) {
        match self.op_type {
            OpType::Add => {
                for i in 0..2 {
                    match &self.inputs[i].grad {
                        Some(g) => {
                            self.inputs[i].grad = Some(g + incoming_grad);
                        }
                        None => {
                            self.inputs[i].grad = Some(incoming_grad.to_owned());
                        }
                    };
                }
            }
            OpType::Sub => {
                for i in 0..2 {
                    match &self.inputs[i].grad {
                        Some(g) => {
                            self.inputs[i].grad = Some(g - incoming_grad);
                        }
                        None => {
                            self.inputs[i].grad = Some(-incoming_grad.to_owned());
                        }
                    };
                }
            }
            OpType::Mul => {
                match &self.inputs[0].grad {
                    Some(g) => {
                        self.inputs[0].grad = Some(g + incoming_grad * &self.inputs[1].data);
                    }
                    None => {
                        self.inputs[0].grad = Some(incoming_grad * &self.inputs[1].data);
                    }
                };

                match &self.inputs[1].grad {
                    Some(g) => {
                        self.inputs[1].grad = Some(g + incoming_grad * &self.inputs[0].data);
                    }
                    None => {
                        self.inputs[1].grad = Some(incoming_grad * &self.inputs[0].data);
                    }
                };
            }
        };
    }
}

#[derive(Debug)]
pub struct Tensor<'a> {
    pub data: Array2<f64>,
    pub grad: Option<Array2<f64>>,
    _ctx: Option<OpCtx<'a>>,
    requires_grad: Option<bool>,
}

impl<'a> Tensor<'a> {
    pub fn new(a: Array2<f64>) -> Tensor<'a> {
        Tensor { data: a, grad: None, _ctx: None, requires_grad: Some(false) }
    }
}

impl<'a> Tensor<'a> {
    pub fn add(&'a mut self, x: &'a mut Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data + &x.data,
            grad: None,
            requires_grad: None,
            _ctx: Some(OpCtx {
                inputs: vec![self, x],
                op_type: OpType::Add,
            }),
        }
    }

    pub fn mul(&'a mut self, x: &'a mut Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data * &x.data,
            grad: None,
            requires_grad: None,
            _ctx: Some(OpCtx {
                inputs: vec![self, x],
                op_type: OpType::Mul,
            }),
        }
    }

    pub fn sub(&'a mut self, x: &'a mut Tensor<'a>) -> Tensor<'a> {
        Tensor {
            data: &self.data - &x.data,
            grad: None,
            requires_grad: None,
            _ctx: Some(OpCtx {
                inputs: vec![self, x],
                op_type: OpType::Sub,
            }),
        }
    }

    pub fn backward(&mut self) {
        let mut seen = HashSet::<*mut Tensor>::new();
        let mut topo = Vec::<&mut Tensor>::new();
        self.grad = Some(Array2::<f64>::ones(self.data.dim()));

        let mut my_stack = Vec::<&mut Tensor>::new();
        my_stack.push(self);

        while my_stack.len() > 0 {
            unsafe {
                let elem = my_stack.pop().unwrap();
                let elem_raw_ptr = elem as *mut Tensor;
                seen.insert(elem_raw_ptr);
                let _ctx = (*elem_raw_ptr)._ctx.as_mut();
                match _ctx {
                    Some(op_node) => {
                        let input_iter = op_node.inputs.iter_mut();
                        if input_iter.len() > 0 {
                            for t in input_iter {
                                let deref_t = t.deref_mut() as *mut Tensor;
                                if !seen.contains(&deref_t) {
                                    my_stack.push(elem_raw_ptr.as_mut().unwrap());
                                    my_stack.push(*t);
                                } else {
                                    topo.push(elem_raw_ptr.as_mut().unwrap());
                                }
                            }
                        } else {
                            topo.push(elem_raw_ptr.as_mut().unwrap());
                        }
                    }
                    None => {
                        topo.push(elem_raw_ptr.as_mut().unwrap());
                    }
                }
            }
        }

        topo.reverse();
        println!("Topo order");
        for elem in topo.iter() {
            println!("{:?}", elem.data);
        }

        self.grad = Some(Array2::<f64>::ones(self.data.dim()));
        for t in topo {
            let incoming_grad = t.grad.as_ref().unwrap();
            match &mut t._ctx {
                Some(op_node) => {
                    op_node.backward(&incoming_grad);
                }
                None => {}
            }
        }
    }
}