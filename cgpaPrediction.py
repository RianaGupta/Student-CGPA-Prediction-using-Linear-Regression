import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n=200 #number of students
data = {
    "StudyHours": np.random.uniform(1, 10, n),
    "Attendance": np.random.uniform(50, 100, n),
    "PreviousCGPA": np.random.uniform(5, 9.5, n),
    "AssignmentScore": np.random.uniform(40, 100, n),
    "SleepHours": np.random.uniform(4, 9, n),
    "ExtracurricularScore": np.random.uniform(0, 10, n)
}

df=pd.DataFrame(data)

# CGPA formula
df["FinalCGPA"] = (
    0.25 * df["StudyHours"] +
    0.02 * df["Attendance"] +
    0.35 * df["PreviousCGPA"] +
    0.015 * df["AssignmentScore"] -
    0.08 * df["SleepHours"] +
    0.05 * df["ExtracurricularScore"] +
    np.random.normal(0, 0.7, n)   # noise
)

df["FinalCGPA"]=df["FinalCGPA"].clip(0,10)

X=df.drop("FinalCGPA",axis=1)
y=df["FinalCGPA"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=45
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))


coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nFeature Importance:")
print(coef_df.sort_values(by="Coefficient", ascending=False))

plt.scatter(y_test, y_pred)
plt.plot([0, 10], [0, 10])
plt.xlabel("Actual CGPA")
plt.ylabel("Predicted CGPA")
plt.title("Actual vs Predicted CGPA")
plt.show()


print("\n--- Student Advisory System ---")
# new student different from dataset whose data is given below
new_student = pd.DataFrame({
    "StudyHours": [5],
    "Attendance": [75],
    "PreviousCGPA": [7],
    "AssignmentScore": [70],
    "SleepHours": [6],
    "ExtracurricularScore": [6]
})

predicted_cgpa = model.predict(new_student)[0]

print("Predicted CGPA:", round(predicted_cgpa, 2))

# Recommendation Logic
if predicted_cgpa < 6.5:
    print("Recommendation: Academic improvement needed. Increase study hours and attendance.")
elif predicted_cgpa < 8:
    print("Recommendation: Performance is average. Maintain consistency and improve assignments.")
else:
    print("Recommendation: Excellent academic standing. Keep up the good work!")
