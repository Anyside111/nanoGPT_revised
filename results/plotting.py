import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv(losses_file_path)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(data['Iteration'], data['Train Loss'], label='Train Loss')
plt.plot(data['Iteration'], data['Validation Loss'], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Iterations')
plt.legend()
plt.show()
