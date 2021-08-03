use crate::matrix::Matrix;

pub struct NeuralNetwork {
    topology: Vec<usize>,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(topology: Vec<usize>) -> Result<NeuralNetwork, &'static str> {
        if topology.len() < 2 {
            return Err("Topology must have an input layer and an output layer.");
        }

        let mut nn = NeuralNetwork {
            topology: topology,
            weights: Vec::new(),
            bias: Vec::new(),
        };

        nn.init_weights();
        nn.randomize_weights();

        nn.init_bias();
        nn.randomize_bias();

        Ok(nn)
    }

    pub fn print_network(&self) {
        print!("Topology: ");
        for i in 0..self.topology.len() {
            print!("{}, ", self.topology[i]);
        }
        println!("");
        println!("");

        println!("Weights:");
        for i in 0..self.weights.len() {
            self.weights[i].print_matrix();
        }
        println!("");

        println!("Bias:");
        for i in 0..self.bias.len() {
            self.bias[i].print_matrix();
        }
        println!("");
    }

    fn init_weights(&mut self) {
        for i in 0..self.topology.len() {
            if i + 1 < self.topology.len() {
                self.weights.push(Matrix::new(self.topology[i], self.topology[i+1]));
            }
        }
    }

    fn randomize_weights(&mut self) {
        for i in 0..self.weights.len() {
            self.weights[i].randomize();
        }
    }

    fn init_bias(&mut self) {
        for i in 0..self.topology.len() {
            if i > 0 {
                self.bias.push(Matrix::new(1, self.topology[i]));
            }
        }
    }

    fn randomize_bias(&mut self) {
        for i in 0..self.bias.len() {
            self.bias[i].randomize();
        }
    }

    pub fn feedforward(&self, input: &Matrix) -> Result<Matrix, &str> {
        if input.rows() != 1 || input.columns() != self.topology[0] {
            return Err("Input does not match the topology.");
        }

        let mut layer: Matrix = Matrix::new(0, 0);
        for i in 0..self.topology.len() - 1 {
            if i == 0 {
                layer = Matrix::multiply(input, &self.weights[i]).unwrap();
            } else {
                layer = Matrix::multiply(&layer, &self.weights[i]).unwrap();
            }
            layer = Matrix::add(&layer, &self.bias[i]).unwrap();
            layer = Matrix::activate(&layer);
        }

        Ok(layer)
    }

    pub fn backpropagation(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64)
        -> Result<(), &str> {
        if input.rows() != 1 || input.columns() != self.topology[0] {
            return Err("Input does not match the topology.");
        }

        let mut a: Vec<Matrix> = Vec::new();
        let mut z: Vec<Matrix> = Vec::new();

        z.push(input.clone());
        a.push(input.clone());

        let mut layer: Matrix = Matrix::new(0, 0);
        for i in 0..self.topology.len() - 1 {
            if i == 0 {
                layer = Matrix::multiply(input, &self.weights[i]).unwrap();
            } else {
                layer = Matrix::multiply(&layer, &self.weights[i]).unwrap();
            }
            layer = Matrix::add(&layer, &self.bias[i]).unwrap();
            z.push(layer.clone());
            layer = Matrix::activate(&layer);
            a.push(layer.clone());
        }

        let mut error_l = Matrix::new(0, 0);

        let mut i: i64 = self.topology.len() as i64 - 2;
        while i >= 0 {
            // Last layer
            if i == self.topology.len() as i64 - 2 {
                let loss_l = Matrix::subtract(target, &a[i as usize + 1]).unwrap();
                error_l = Matrix::hadamard_product(
                    &loss_l,
                    &Matrix::activate_prime(&z[i as usize + 1])
                ).unwrap();
            // all other layers
            } else {
                let weights_t = Matrix::transpose(&self.weights[i as usize + 1]);
                error_l = Matrix::hadamard_product(
                    &Matrix::multiply(
                        &error_l,
                        &weights_t
                    ).unwrap(),
                    &Matrix::activate_prime(&z[i as usize + 1])
                ).unwrap();
            }

            self.bias[i as usize] = Matrix::add(
                &self.bias[i as usize],
                &Matrix::scalar(&error_l, learning_rate),
            ).unwrap();

            for k in 0..self.weights[i as usize].rows() {
                for j in 0..self.weights[i as usize].columns() {
                    let q: f64;
                    let e: f64;
                    match a[i as usize].get_value(0, k) {
                        Some(x) => q = x,
                        None => return Err("Could not get a value."),
                    }
                    match error_l.get_value(0, j) {
                        Some(x) => e = x,
                        None => return Err("Could not get e value."),
                    }
                    match self.weights[i as usize].get_value(k, j) {
                        Some(x) => {
                            let value = x + learning_rate * q * e;
                            self.weights[i as usize].set_value(k, j, value).unwrap();
                        },
                        None => return Err("Could not compute new weight values."),
                    }
                }
            }
            i -= 1;
        }

        Ok(())
    }
}
