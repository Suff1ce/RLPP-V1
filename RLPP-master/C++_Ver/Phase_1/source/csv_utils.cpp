#include "csv_utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

Eigen::MatrixXd load_csv_matrix(const std::string&path){ // Creates a function that reads a CSV file and return an Eigen matrix
    std::ifstream file(path);
    if (!file.is_open()){
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    std::size_t expected_cols = 0;

    while (std::getline(file, line)){ // Read each line from the file, store in variable "line"
        if (line.empty()){
            continue; // Skip empty lines
        }
        std::stringstream line_stream(line);
        std::string cell;
        std::vector<double> row;
        
        while (std::getline(line_stream, cell, ',')){
            row.push_back(std::stod(cell)); // Convert the "cell" string to a double and add to the current row
        }

        if (row.empty()){
            continue; // Skip rows that are empty after parsing
        }

        if (expected_cols ==0){
            expected_cols = row.size(); // Set expected column count based on the first non-empty row
        }
        else if (row.size() != expected_cols){
            throw std::runtime_error("Inconsistent number of columns in file: " + path);
        }

        rows.push_back(row); // Add the parsed row to the list of rows, forming a 2D array

    }

    if (rows.empty()){
        throw std::runtime_error("CSV file is empty, no data found: " + path);
    }

    Eigen::MatrixXd matrix(static_cast<int>(rows.size()), static_cast<int>(expected_cols));

    for (int i = 0; i < rows.size(); i++){
        for (int j = 0; j < expected_cols; j++){
            matrix(i,j) = rows[i][j]; // Fill the Eigen matrix with the parsed values from the 2D vector
        }
    }

return matrix;

}

Eigen::VectorXd load_csv_vector(const std::string& path) {
    Eigen::MatrixXd M = load_csv_matrix(path);
    if (M.cols() == 1) {
        return M.col(0);
    }
    if (M.rows() == 1) {
        return M.row(0).transpose();
    }
    throw std::runtime_error("Expected vector CSV (Nx1 or 1xN): " + path);
}

void save_csv_matrix(const std::string& path, const Eigen::MatrixXd& matrix){ // Creates a function that saves an Eigen matrix to a CSV file
    std::ofstream file(path);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file for writing: " + path);
    }

    for(int i = 0; i < matrix.rows(); i++){
        for(int j = 0; j < matrix.cols(); j++){
            file << matrix(i,j);
            if(j < matrix.cols() - 1){
                file << ",";
            }
        }
        file << "\n"; // Newline after each row
    }
}

void save_csv_vector(const std::string& path, const Eigen::VectorXd& vector){
    std::ofstream file(path);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file for writing: " + path);
    }

    // Write as a single-column CSV (one value per line).
    for (int i = 0; i < vector.size(); ++i) {
        file << vector(i) << "\n";
    }
}