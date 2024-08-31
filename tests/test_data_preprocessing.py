import unittest
import pandas as pd
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        sample_data = {
            'timestamp': ['2024-08-01 00:00:00', '2024-08-01 01:00:00'],
            'temperature': [30.5, 30.0],
            'humidity': [70, 72],
            'energy_consumption': [150.2, 152.3]
        }
        df = pd.DataFrame(sample_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        processed_df = preprocess_data(df)
        
        self.assertEqual(processed_df.isnull().sum().sum(), 0, "Data should have no missing values")
        self.assertIn('hour', processed_df.columns, "Hour feature should be present")
        self.assertIn('temperature', processed_df.columns, "Temperature feature should be present")

if __name__ == '__main__':
    unittest.main()
