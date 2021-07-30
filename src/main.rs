pub mod nn;
pub mod matrix;

use crate::nn::NeuralNetwork;
use crate::matrix::Matrix;

fn main() {
    let nn = NeuralNetwork::new(vec![4, 2, 1]).unwrap();
    nn.print_network();

    let mut input = Matrix::new(1, 4);
    input.set_matrix(vec![vec![2.8, 5.6, 8.6, 0.2]]).unwrap();

    println!("Output:");
    let output = nn.feedforward(&input).unwrap();
    output.print_matrix();

    println!("Matrix Transpose:");
    let mut a = Matrix::new(3, 2);
    a.set_matrix(vec![
        vec![1.0, -2.0],
        vec![3.0, 0.0],
        vec![7.0, 5.0],
    ]).unwrap();
    a.print_matrix();

    let t = Matrix::transpose(&a);
    t.print_matrix();
}
