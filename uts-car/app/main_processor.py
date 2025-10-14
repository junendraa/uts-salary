from src.data_processor import DataProcessor

# Inisialisasi kelas
processor = DataProcessor()

# 1️⃣ Load data dari direktori kamu
df = processor.load_data("data/raw/data.csv")

# 2️⃣ Bersihkan data
df_clean = processor.clean_data(df)

# 3️⃣ Lakukan feature engineering
df_feat = processor.feature_engineering(df_clean)

# 4️⃣ Siapkan fitur dan target
X, y, feature_names = processor.prepare_features(df_feat, target_col='selling_price')

print("\n🎯 Data preprocessing selesai!")
print(f"Feature matrix (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Fitur digunakan: {feature_names[:10]} ... total {len(feature_names)} fitur")
