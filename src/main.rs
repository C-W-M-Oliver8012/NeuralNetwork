pub mod nn;
pub mod matrix;

use crate::nn::NeuralNetwork;
use crate::matrix::Matrix;

fn main() {
    let nn = NeuralNetwork::new(vec![2, 2, 1]).unwrap();
    nn.print_network();

    let mut input = Matrix::new(1, 2);
    input.set_matrix(vec![vec![2.8, 5.6]]).unwrap();

    let output = nn.feedforward(&input).unwrap();
    output.print_matrix();
}
