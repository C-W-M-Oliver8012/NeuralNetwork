pub mod nn;
pub mod matrix;

use crate::nn::NeuralNetwork;
use crate::matrix::Matrix;

fn main() {
    let mut nn = NeuralNetwork::new(vec![2, 4, 4, 1]).unwrap();

    let mut inputs: Vec<Matrix> = Vec::new();
    let mut targets: Vec<Matrix> = Vec::new();
    for _i in 0..4 {
        inputs.push(Matrix::new(1, 2));
        targets.push(Matrix::new(1, 1));
    }

    inputs[0].set_matrix(vec![vec![0.0, 0.0]]).unwrap();
    inputs[1].set_matrix(vec![vec![1.0, 0.0]]).unwrap();
    inputs[2].set_matrix(vec![vec![0.0, 1.0]]).unwrap();
    inputs[3].set_matrix(vec![vec![1.0, 1.0]]).unwrap();

    targets[0].set_matrix(vec![vec![0.0]]).unwrap();
    targets[1].set_matrix(vec![vec![1.0]]).unwrap();
    targets[2].set_matrix(vec![vec![1.0]]).unwrap();
    targets[3].set_matrix(vec![vec![0.0]]).unwrap();

    for _i in 0..90000 {
        for j in 0..4 {
            nn.backpropagation(&inputs[j], &targets[j], 0.001).unwrap();
        }
    }

    nn.print_network();

    for i in 0..4 {
        println!("Inputs:");
        inputs[i].print_matrix();

        println!("Outputs:");
        let output = nn.feedforward(&inputs[i]).unwrap();
        output.print_matrix();
    }
}
