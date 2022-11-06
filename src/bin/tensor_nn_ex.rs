use tinygrad_rust::{datasets::{mnist, Dataloader}, tensor::Tensor};

fn main() {
    let w1 = Tensor::randn([30, 28 * 28]);
    let b1 = Tensor::randn([30, 1]);

    let w2 = Tensor::randn([10, 30]);
    let b2 = Tensor::randn([10, 1]);

    let dataloader = mnist::load_data();

    // Feed forward with backprop for a particular case
    let (_x, _y) = dataloader.get_by_idx(0);
    let x = Tensor::new(_x);
    let y = Tensor::new(_y);

    // Layer 1
    let w1x = w1.matmul(&x);
    let z1 = w1x.add(&b1);
    let a1 = z1.sigmoid();

    // Layer 2
    let w2a1 = w2.matmul(&a1);
    let z2 = w2a1.add(&b2);
    let out = z2.sigmoid();

    // Calculate loss
    let diff = out.sub(&y);
    let cost = diff.sq();
    let loss = cost.mean();

    //Backprop
    loss.backward();

    println!("gradW1 {:?}", w1.grad.borrow().as_ref().unwrap());
    println!("gradW2 {:?}", w2.grad.borrow().as_ref().unwrap());
    println!("gradB1 {:?}", b1.grad.borrow().as_ref().unwrap());
    println!("gradB2 {:?}", b2.grad.borrow().as_ref().unwrap());
}
