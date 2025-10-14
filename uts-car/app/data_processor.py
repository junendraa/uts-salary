import numpy as np
import pandas as pd
import io

class DataProcessor:
    def __init__(self, file=None):
        """
        Bisa menerima:
        - path string ('data/raw/data.csv')
        - file upload dari Flask (request.files['file'])
        """
        self.filepath = None
        self.df = None
        self.feature_names = None

        if file is not None:
            if isinstance(file, str):
                # kasus: file path biasa
                self.filepath = file
                self.df = pd.read_csv(file)
            else:
                # kasus: file upload dari web Flask
                self.df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))

    def process(self):
        """
        Load dan process data salary.
        Kalau kolom sesuai dengan project salary ('age', 'years_exp', 'gender', 'education', 'salary'),
        maka gunakan mode khusus. Kalau tidak, gunakan fallback numerik otomatis.
        """
        if self.df is None:
            raise ValueError("Data belum dimuat. Pastikan file CSV valid.")

        df = self.df.copy()

        # Cek apakah kolom salary dataset kamu ada
        expected_cols = ['age', 'years_exp', 'gender', 'education', 'salary']
        if all(col in df.columns for col in expected_cols):
            # Mode: dataset salary utama
            feature_cols = ['age', 'years_exp', 'gender', 'education']
            target_col = 'salary'

            X = df[feature_cols].values.astype(float)
            y = df[target_col].values.astype(float)

            # Hapus NaN
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]

            self.feature_names = feature_cols

            print(f"\n✓ Data processed (Salary Dataset):")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {len(feature_cols)}")
            print(f"  Feature names: {feature_cols}")
            print(f"  Shape X: {X.shape}")
            print(f"  Shape y: {y.shape}")
            print(f"  Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
            print(f"  Salary mean: ${y.mean():,.0f}")

            return X, y, self.feature_names

        else:
            # Mode fallback: auto detect numeric
            df = df.select_dtypes(include=["number"]).dropna()
            if df.shape[1] < 2:
                raise ValueError("Dataset harus memiliki minimal 1 fitur dan 1 target numerik.")

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            self.feature_names = df.columns[:-1].tolist()

            print(f"\n✓ Data processed (Generic Numeric Dataset):")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Shape X: {X.shape}")
            print(f"  Shape y: {y.shape}")

            return X, y, self.feature_names
