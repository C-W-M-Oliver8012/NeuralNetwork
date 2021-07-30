use rand;

pub struct Matrix {
    rows: usize,
    columns: usize,
    matrix: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        let mut matrix = Matrix {
            rows: rows,
            columns: columns,
            matrix: Vec::new(),
        };

        for i in 0..rows {
            matrix.matrix.push(Vec::new());
            for _j in 0..columns {
                matrix.matrix[i].push(0.0);
            }
        }

        matrix
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn set_matrix(&mut self, matrix: Vec<Vec<f64>>) -> Result<(), &str> {
        if matrix.len() != self.rows {
            return Err("Matrix is wrong size.");
        }

        for i in 0..matrix.len() {
            if matrix[i].len() != self.columns {
                return Err("Matrix is wrong size.");
            }
        }

        for i in 0..self.rows {
            for j in 0..self.columns {
                self.matrix[i][j] = matrix[i][j];
            }
        }

        Ok(())
    }

    pub fn set_value(&mut self, row: usize, column: usize, value: f64) -> Result<(), &str> {
        if row >= self.rows || column >= self.columns {
            return Err("Row or column are not within the matrix.");
        }

        self.matrix[row][column] = value;

        Ok(())
    }

    pub fn get_matrix(&self) -> Vec<Vec<f64>> {
        self.matrix.clone()
    }

    pub fn get_value(&self, row: usize, column: usize) -> Option<f64> {
        if row >= self.rows || column >= self.columns {
            return None
        }

        Some(self.matrix[row][column])
    }

    pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        if a.rows != b.rows || a.columns != b.columns {
            return Err("Matrixes are not the same size.");
        }

        let mut m = Matrix::new(a.rows, a.columns);

        for i in 0..a.rows {
            for j in 0..a.columns {
                m.matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
            }
        }

        Ok(m)
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        if a.columns != b.rows {
            return Err("Matrixes cannot be multiplied.");
        }

        let mut m = Matrix::new(a.rows, b.columns);

        for i in 0..a.rows {
            for j in 0..a.columns {
                for l in 0..b.columns {
                        m.matrix[i][l] += a.matrix[i][j] * b.matrix[j][l];
                }
            }
        }
        Ok(m)
    }

    pub fn randomize(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.matrix[i][j] = rand::random::<f64>() * 2.0 - 1.0;
            }
        }
    }

    pub fn print_matrix(&self) {
        println!("[");
        for i in 0..self.rows {
            print!(" [");
            for j in 0..self.columns {
                print!("{}, ", self.matrix[i][j]);
            }
            println!("],");
        }
        println!("]");
        println!("");
    }

    pub fn activate(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.matrix[i][j] = leaky_relu(self.matrix[i][j]);
            }
        }
    }
}

pub fn leaky_relu(x: f64) -> f64 {
    if x >= 0.0 {
        return x;
    } else {
        return x * 0.1;
    }
}
