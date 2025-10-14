import numpy as np
import pandas as pd
import os

class MatrixOperations:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)

    def print_matrix(self, name="Matrix", max_rows=10):
        print(f"\n{name}")
        try:
            rows, cols = self.matrix.shape
            print(f"Shape: ({rows}, {cols})")
        except:
            print("⚠️ Matrix tidak valid")
            return

        if len(self.matrix) > max_rows:
            print(self.matrix[:max_rows])
            print(f"... ({len(self.matrix) - max_rows} more rows)")
        else:
            print(self.matrix)

    def transpose(self):
        """Transpose matrix"""
        return MatrixOperations(self.matrix.T)

    def inverse(self):
        """Inverse matrix (hanya untuk matrix persegi)"""
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix tidak persegi, tidak bisa inverse.")
        inv_matrix = np.linalg.inv(self.matrix)
        return MatrixOperations(inv_matrix)

    @staticmethod
    def import_from_csv(filepath: str, has_header: bool = True) -> 'MatrixOperations':
        """Import matrix dari CSV file"""
        try:
            df = pd.read_csv(filepath, header=0 if has_header else None, on_bad_lines='skip')
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("Tidak ada kolom numerik di CSV.")
            matrix_data = numeric_df.to_numpy()
            print(f"✅ Berhasil import matrix dari {filepath}")
            print(f"   Shape: {matrix_data.shape}")
            return MatrixOperations(matrix_data)
        except Exception as e:
            print(f"❌ Error importing CSV: {e}")
            return MatrixOperations(np.array([]))


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("============================================================")
    print("TESTING MATRIX OPERATIONS")
    print("============================================================\n")

    # Path ke data CSV
    file_path = os.path.join(os.path.dirname(__file__), "../data/raw/data.csv")

    print("[TEST 1] IMPORT MATRIX DARI CSV")
    C = MatrixOperations.import_from_csv(file_path)

    if C.matrix.size == 0:
        print("❌ Matrix kosong")
    else:
        C.print_matrix("Matrix dari CSV", max_rows=10)

        # TRANSPOSE
        print("\nTranspose dari data CSV")
        C_T = C.transpose()
        C_T.print_matrix("Transpose Matrix CSV", max_rows=5)

        # INVERSE
        try:
            if C.matrix.shape[0] == C.matrix.shape[1]:
                C_inv = C.inverse()
                C_inv.print_matrix("Inverse Matrix CSV", max_rows=5)
            else:
                print("\n⚠️ Matrix tidak persegi — menggunakan dummy matrix untuk inverse.")
                dummy = np.array([[4, 7], [2, 6]])
                D = MatrixOperations(dummy)
                D_inv = D.inverse()
                D_inv.print_matrix("Inverse Dummy Matrix (contoh)", max_rows=5)
        except Exception as e:
            print(f"⚠️ Error inverse: {e}")

    print("\n✅ Testing selesai.")
