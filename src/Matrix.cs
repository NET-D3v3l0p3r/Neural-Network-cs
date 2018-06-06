// Port: Guy Ross
// Original: Daniel Shiffman
//
// 1/ 17/ 18
//
// Neural Network Direct Port from Daniel Shiffmans Neural Network Library.
// Ported into c# from p5 js.

using System;
using System.Collections.Generic;

namespace CSharpNeuralNetwork
{
    public class Matrix
    {
        public Int32 Rows { get; private set; }
        public Int32 Cols {  get; private set; }
        public List<Double[]> Values { get; private set; }

        public Matrix(Int32 rows, Int32 cols)
        {
            Rows = rows;
            Cols = cols;
            Values = new List<Double[]>();
            
            CreateValues(); 
            
            // Without "CreateValues" this will give an exception, because Values has no items in it
            ZeroValues();
        }
        
        private void CreateValues()
        {
            for(int i = 0; i < Rows; i++)
            {
                Values.Add(new Double[Cols]);
            }
        }

        public void ZeroValues()
        {
 
            for (int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Cols; j++)
                {
                    Values[i][j] = 0;
                }
            }
        }

        public void Randomize()
        {
            for(int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Cols; j++)
                {
                    Values[i][j] = new Random().Next(-1, 1);
                }
            }
        }

        public List<Double> ToList()
        {
            var lst = new List<double>();

            for(int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Cols; j++)
                {
                    lst.Add(Values[i][j]);
                }
            }

            return lst;
        }

        public Matrix Transpose()
        {
            
            // Has problems.. Values gives out of bounds exception, which is logically because results.Rows is Values cols...
            var result = new Matrix(Cols, Rows);

            for(int i = 0; i < result.Rows; i++)
            {
                for(int j = 0; j < result.Cols; j++)
                {
                    result.Values[i][j] = Values[i][j];
                }
            }

            return result;
        }
    
        public Matrix Copy()
        {
            var result = new Matrix(Rows, Cols);
            for(int i = 0; i < result.Rows; i++)
            {
                for(int j = 0; j < result.Cols; j++)
                {
                    result.Values[i][j] = Values[i][j];
                }
            }

            return result;
        }

        public void Add(Matrix value)
        {
            for(int i = 0; i < Rows; i++)
            {
                for(int j = 0; j < Cols; j++)
                {
                    Values[i][j] += value.Values[i][j];
                }
            }
        }

        public void Add(Int32 value)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Values[i][j] += value;
                }
            }
        }

        public void Multiply(Matrix value)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Values[i][j] *= value.Values[i][j];
                }
            }
        }

        public void Multiply(Double value)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Values[i][j] *= value;
                }
            }
        }

        public void Subtract(Matrix value)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Values[i][j] -= value.Values[i][j];
                }
            }
        }

        public void Subtract(Int32 value)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Values[i][j] -= value;
                }
            }
        }

        #region Static Methods
        public static Matrix Map(Matrix matrix, Func<Double, Double> function)
        {
            var result = new Matrix(matrix.Rows, matrix.Cols);

            for(int i = 0; i < result.Rows; i++)
            {
                for(int j = 0; j < result.Cols; j++)
                {
                    result.Values[i][j] = function(matrix.Values[i][j]);
                }
            }

            return result;
        }

        public static Matrix MultiplyMatrices(Matrix first, Matrix second)
        {
            
            // Somewhere this will also return an error.. probably because of "for(int k = 0; *j* < first.Cols; k++)"
            // For an better approach see https://pastebin.com/0dUuttDA all possiblities are handled _ 
            if(first.Cols != second.Rows)
            {
                throw new Exception("Incompatible Matrix Size");
            }

            var result = new Matrix(first.Rows, second.Cols);

            for(int i = 0; i < first.Rows; i++)
            {
                for(int j = 0; j < second.Cols; j++)
                {
                    double sum = 0.0; 
                    for(int k = 0; j < first.Cols; k++) // why j?
                    {
                        sum += first.Values[i][k] * second.Values[k][j];
                    }

                    result.Values[i][j] = sum;
                }
            }

            return result;
        }

        public static Matrix FromList(List<Double> list)
        {
            var matrix = new Matrix(list.Count, 1);

            for(int i = 0; i < list.Count; i++)
            {
                matrix.Values[i][0] = list[i];
            }
            return matrix;
        }

        public static Matrix SubtractMatrices(Matrix first, Matrix second)
        {
            var result = new Matrix(first.Rows, first.Cols);
            for(int i = 0; i < result.Rows; i++)
            {
                for(int j = 0; j < result.Cols; j++)
                {
                    result.Values[i][j] = first.Values[i][j] - second.Values[i][j];
                }
            }
            return result;
        }
        #endregion
    }
}
